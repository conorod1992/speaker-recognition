"""Speaker recognition module."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientTimeout
from homeassistant.components import media_source
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .audio import decode_wav

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ADDON_URL = "http://localhost:8099"


class RecognitionBackendUnavailable(RuntimeError):
    """Raised when the Speaker Recognition backend cannot provide valid status."""


@dataclass(frozen=True)
class TrainingResult:
    """Training response returned by the Speaker Recognition app."""

    users_trained: list[str]
    profile_consistency: dict[str, float] = field(default_factory=dict)
    outlier_samples: dict[str, list[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecognitionResult:
    """Recognition response returned by the Speaker Recognition app."""

    user_id: str | None
    candidate_user_id: str
    confidence: float
    similarity: float
    margin: float | None
    accepted: bool
    all_scores: dict[str, float]


class SpeakerRecognition:
    """Handle speaker recognition from audio data."""

    def __init__(
        self,
        hass: HomeAssistant,
        voice_samples: list[dict],
        base_url: str = DEFAULT_ADDON_URL,
    ) -> None:
        """Initialize speaker recognition.

        Args:
            hass: Home Assistant instance
            voice_samples: List of voice samples with user and audio file info
            base_url: Base URL of the speaker recognition service
        """
        self.hass = hass
        self.voice_samples = voice_samples
        self._trained = False
        self._enrolled_users: list[str] = []
        self._base_url = base_url.rstrip("/")

    @property
    def enrolled_users(self) -> list[str]:
        """Return users currently reported as enrolled by the backend."""
        return list(self._enrolled_users)

    async def _async_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call the local Speaker Recognition app without external dependencies."""
        session = async_get_clientsession(self.hass)
        async with session.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=ClientTimeout(total=300),
        ) as response:
            response.raise_for_status()
            data = await response.json()

        if not isinstance(data, dict):
            raise ValueError("Unexpected response from Speaker Recognition app")
        return data

    async def _async_get(self, path: str) -> dict[str, Any]:
        """Read lightweight status from the local Speaker Recognition app."""
        session = async_get_clientsession(self.hass)
        async with session.get(
            f"{self._base_url}{path}", timeout=ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            data = await response.json()

        if not isinstance(data, dict):
            raise ValueError("Unexpected response from Speaker Recognition app")
        return data

    async def _async_read_media(self, media_content_id: str) -> bytes:
        """Resolve and read a locally selected media-source item."""
        resolved_media = await media_source.async_resolve_media(
            self.hass, media_content_id, None
        )
        if resolved_media.path is None:
            raise ValueError("Selected media source does not provide a local file")
        return await self.hass.async_add_executor_job(resolved_media.path.read_bytes)

    @staticmethod
    def _parse_training_response(
        response: dict[str, Any], expected_users: set[str]
    ) -> TrainingResult:
        """Validate a backend training response and retain profile diagnostics."""
        trained_users = response.get("trained_users")
        if not isinstance(trained_users, list) or not all(
            isinstance(user, str) for user in trained_users
        ):
            raise ValueError("Invalid training response from Speaker Recognition app")
        if not trained_users:
            raise ValueError("Speaker Recognition app did not train any users")

        trained_user_set = set(trained_users)
        if not expected_users.issubset(trained_user_set):
            missing_users = sorted(expected_users - trained_user_set)
            raise ValueError(
                "Speaker Recognition app did not train requested users: "
                + ", ".join(missing_users)
            )

        raw_consistency = response.get("profile_consistency", {})
        raw_outliers = response.get("outlier_samples", {})
        if not isinstance(raw_consistency, dict) or not isinstance(raw_outliers, dict):
            raise ValueError("Invalid profile diagnostics from Speaker Recognition app")

        profile_consistency: dict[str, float] = {}
        for user, value in raw_consistency.items():
            if not isinstance(user, str) or not isinstance(value, (int, float)):
                raise ValueError("Invalid profile consistency from Speaker Recognition app")
            profile_consistency[user] = float(value)

        outlier_samples: dict[str, list[int]] = {}
        for user, values in raw_outliers.items():
            if (
                not isinstance(user, str)
                or not isinstance(values, list)
                or not all(isinstance(index, int) and index > 0 for index in values)
            ):
                raise ValueError("Invalid profile outliers from Speaker Recognition app")
            outlier_samples[user] = list(values)

        return TrainingResult(
            users_trained=trained_users,
            profile_consistency=profile_consistency,
            outlier_samples=outlier_samples,
        )

    async def async_refresh_status(self) -> bool:
        """Refresh recognition availability from persisted backend profiles."""
        try:
            response = await self._async_get("/health")
            trained = response.get("trained")
            enrolled_users = response.get("enrolled_users")
            if not isinstance(trained, bool) or not isinstance(enrolled_users, list):
                raise ValueError("Invalid profile status from Speaker Recognition app")
            if not all(isinstance(user, str) for user in enrolled_users):
                raise ValueError(
                    "Invalid enrolled user list from Speaker Recognition app"
                )
        except (ClientError, OSError, ValueError, TypeError) as error:
            self._trained = False
            self._enrolled_users = []
            _LOGGER.warning("Unable to read speaker recognition status: %s", error)
            raise RecognitionBackendUnavailable(
                "Speaker Recognition backend is unavailable"
            ) from error

        self._enrolled_users = sorted(enrolled_users)
        self._trained = trained and bool(enrolled_users)
        _LOGGER.info(
            "Speaker recognition backend has %d persisted profiles",
            len(enrolled_users),
        )
        return self._trained

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        """Train configured samples for selected newly enrolled users."""
        selected_samples = [
            sample
            for sample in self.voice_samples
            if user_ids is None or sample.get("user") in user_ids
        ]
        _LOGGER.debug(
            "Training speaker recognition with %d voice samples",
            len(selected_samples),
        )

        if not selected_samples:
            _LOGGER.warning("No changed voice samples available for training")
            return False

        expected_users = {
            user
            for sample in selected_samples
            if isinstance((user := sample.get("user")), str)
        }

        try:
            voice_sample_models = []
            for sample in selected_samples:
                user_id = sample["user"]
                selected_media = sample["samples"]
                media_items = (
                    selected_media
                    if isinstance(selected_media, list)
                    else [selected_media]
                )

                for media_item in media_items:
                    if not isinstance(media_item, dict):
                        _LOGGER.warning("Invalid media selection for user: %s", user_id)
                        continue
                    media_id = media_item.get("media_content_id", "")
                    if not isinstance(media_id, str) or not media_id.startswith(
                        "media-source://"
                    ):
                        _LOGGER.warning(
                            "Unsupported media_content_id format: %s", media_id
                        )
                        continue

                    audio_data = await self._async_read_media(media_id)
                    pcm_data, sample_rate = await self.hass.async_add_executor_job(
                        decode_wav, audio_data
                    )
                    audio_base64 = base64.b64encode(pcm_data).decode("utf-8")
                    voice_sample_models.append(
                        {
                            "user": user_id,
                            "audio": {
                                "audio_data": audio_base64,
                                "sample_rate": sample_rate,
                            },
                        }
                    )

            if not voice_sample_models:
                _LOGGER.warning("No valid training samples prepared")
                return False

            response = await self._async_post(
                "/train", {"voice_samples": voice_sample_models}
            )
            result = self._parse_training_response(response, expected_users)

        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.error(
                "Speaker recognition training failed; existing profiles remain usable: %s",
                error,
            )
            return False
        else:
            self._trained = True
            self._enrolled_users = sorted(
                set(self._enrolled_users).union(result.users_trained)
            )
            _LOGGER.info(
                "Speaker recognition training completed: %d users trained",
                len(result.users_trained),
            )
            return True

    async def async_train_pcm_samples(
        self, user_id: str, samples: list[bytes], sample_rate: int = 16000
    ) -> TrainingResult:
        """Train one user from in-memory mono 16-bit PCM enrollment samples."""
        if not samples:
            raise ValueError("No enrollment samples provided")
        payload = {
            "voice_samples": [
                {
                    "user": user_id,
                    "audio": {
                        "audio_data": base64.b64encode(sample).decode("utf-8"),
                        "sample_rate": sample_rate,
                    },
                }
                for sample in samples
            ]
        }
        response = await self._async_post("/train", payload)
        result = self._parse_training_response(response, {user_id})
        self._trained = True
        self._enrolled_users = sorted(set(self._enrolled_users).union({user_id}))
        return result

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        """Recognize speaker from audio data.

        Args:
            audio_data: Raw audio data to analyze (PCM 16-bit)
            sample_rate: Audio sample rate

        Returns:
            RecognitionResult if the backend returns a valid decision, None otherwise
        """
        if not self._trained:
            _LOGGER.warning(
                "Speaker recognition is not trained; skipping recognition request"
            )
            return None

        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            request_started = perf_counter()
            try:
                response = await self._async_post(
                    "/recognize",
                    {
                        "audio": {
                            "audio_data": audio_base64,
                            "sample_rate": sample_rate,
                        }
                    },
                )
            finally:
                _LOGGER.debug(
                    "Speaker recognition backend request completed in %.3fs",
                    perf_counter() - request_started,
                )

            raw_user_id = response.get("user_id")
            confidence = response.get("confidence")
            all_scores = response.get("all_scores")
            candidate_user_id = response.get("candidate_user_id", raw_user_id)
            similarity = response.get("similarity", confidence)
            margin = response.get("margin")
            accepted = response.get("accepted", isinstance(raw_user_id, str))

            if (
                raw_user_id is not None
                and not isinstance(raw_user_id, str)
                or not isinstance(candidate_user_id, str)
                or not isinstance(confidence, (int, float))
                or not isinstance(similarity, (int, float))
                or margin is not None
                and not isinstance(margin, (int, float))
                or not isinstance(accepted, bool)
                or not isinstance(all_scores, dict)
                or not all(
                    isinstance(user, str) and isinstance(score, (int, float))
                    for user, score in all_scores.items()
                )
            ):
                raise ValueError(
                    "Invalid recognition response from Speaker Recognition app"
                )
            if accepted and raw_user_id is None:
                raise ValueError("Accepted recognition result has no user_id")
            if not accepted and raw_user_id is not None:
                raise ValueError("Rejected recognition result unexpectedly has a user_id")

            result = RecognitionResult(
                user_id=raw_user_id,
                candidate_user_id=candidate_user_id,
                confidence=float(confidence),
                similarity=float(similarity),
                margin=float(margin) if margin is not None else None,
                accepted=accepted,
                all_scores={user: float(score) for user, score in all_scores.items()},
            )

        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during recognition: %s", error)
            return None
        else:
            _LOGGER.debug(
                "Recognition result: candidate=%s similarity=%.3f margin=%s accepted=%s",
                result.candidate_user_id,
                result.similarity,
                f"{result.margin:.3f}" if result.margin is not None else "n/a",
                result.accepted,
            )

            return result

    def update_voice_samples(self, voice_samples: list[dict]) -> None:
        """Update configured voice sample media references.

        Args:
            voice_samples: New list of voice samples
        """
        self.voice_samples = voice_samples
        _LOGGER.debug("Configured voice sample references updated")
