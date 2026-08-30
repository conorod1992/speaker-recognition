"""Speaker recognition module."""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientTimeout
from homeassistant.components import media_source
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .audio import decode_wav, read_bounded_wav

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ADDON_URL = "http://localhost:8099"
RECOGNITION_TIMEOUT_SECONDS = 4.0
MIN_PROFILE_SAMPLES = 3


class RecognitionBackendUnavailable(RuntimeError):
    """Raised when the Speaker Recognition backend cannot provide valid status."""


@dataclass(frozen=True)
class TrainingResult:
    """Training response returned by the Speaker Recognition app."""

    users_trained: list[str]


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


@dataclass(frozen=True)
class DenoiseResult:
    """Neural denoise response returned by the Speaker Recognition app."""

    audio_data: bytes
    sample_rate: int
    processing_seconds: float
    engine: str


class SpeakerRecognition:
    """Handle speaker recognition from audio data."""

    def __init__(
        self,
        hass: HomeAssistant,
        voice_samples: list[dict],
        base_url: str = DEFAULT_ADDON_URL,
        api_token: str = "",
    ) -> None:
        """Initialize speaker recognition."""
        self.hass = hass
        self.voice_samples = voice_samples
        self._trained = False
        self._enrolled_users: set[str] = set()
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token.strip()

    def _request_headers(self) -> dict[str, str]:
        """Return authorization headers for a secured remote backend."""
        if not self._api_token:
            return {}
        return {"Authorization": f"Bearer {self._api_token}"}

    @property
    def configured_users(self) -> set[str]:
        """Return currently configured HA user IDs with enrollment samples."""
        return {
            user_id
            for sample in self.voice_samples
            if isinstance((user_id := sample.get("user")), str) and user_id
        }

    @property
    def enrolled_users(self) -> set[str]:
        """Return the backend profiles observed by the latest lifecycle operation."""
        return set(self._enrolled_users)

    async def _async_post(
        self,
        path: str,
        payload: dict[str, Any],
        timeout_seconds: float = 300,
    ) -> dict[str, Any]:
        """Call the local Speaker Recognition app without external dependencies."""
        session = async_get_clientsession(self.hass)
        async with session.post(
            f"{self._base_url}{path}",
            json=payload,
            headers=self._request_headers(),
            timeout=ClientTimeout(total=timeout_seconds),
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
            f"{self._base_url}{path}",
            headers=self._request_headers(),
            timeout=ClientTimeout(total=10),
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
        return await self.hass.async_add_executor_job(read_bounded_wav, resolved_media.path)

    async def async_refresh_status(self) -> bool:
        """Refresh recognition availability from persisted backend profiles."""
        try:
            response = await self._async_get("/health")
            trained = response.get("trained")
            enrolled_users = response.get("enrolled_users")
            if not isinstance(trained, bool) or not isinstance(enrolled_users, list):
                raise ValueError("Invalid profile status from Speaker Recognition app")
            if not all(isinstance(user, str) and user for user in enrolled_users):
                raise ValueError("Invalid enrolled user list from Speaker Recognition app")
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            self._trained = False
            self._enrolled_users = set()
            _LOGGER.warning("Unable to read speaker recognition status: %s", error)
            raise RecognitionBackendUnavailable(
                "Speaker Recognition backend is unavailable"
            ) from error

        self._enrolled_users = set(enrolled_users)
        self._trained = trained and bool(self.configured_users & self._enrolled_users)
        _LOGGER.info(
            "Speaker recognition backend has %d persisted profiles",
            len(self._enrolled_users),
        )
        return self._trained

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        """Train configured samples for selected users."""
        selected_samples = [
            sample
            for sample in self.voice_samples
            if user_ids is None or sample.get("user") in user_ids
        ]
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
                media_items = selected_media if isinstance(selected_media, list) else [selected_media]
                for media_item in media_items:
                    if not isinstance(media_item, dict):
                        continue
                    media_id = media_item.get("media_content_id", "")
                    if not isinstance(media_id, str) or not media_id.startswith("media-source://"):
                        continue
                    audio_data = await self._async_read_media(media_id)
                    pcm_data, sample_rate = await self.hass.async_add_executor_job(
                        decode_wav, audio_data
                    )
                    voice_sample_models.append(
                        {
                            "user": user_id,
                            "audio": {
                                "audio_data": base64.b64encode(pcm_data).decode("utf-8"),
                                "sample_rate": sample_rate,
                            },
                        }
                    )
            if not voice_sample_models:
                return False

            response = await self._async_post("/train", {"voice_samples": voice_sample_models})
            trained_users = response.get("trained_users")
            accepted_samples = response.get("accepted_samples")
            if not isinstance(trained_users, list) or not all(
                isinstance(user, str) for user in trained_users
            ):
                raise ValueError("Invalid training response from Speaker Recognition app")
            if not isinstance(accepted_samples, dict):
                raise ValueError("Training response omitted accepted sample diagnostics")
            trained_user_set = set(trained_users)
            if not expected_users.issubset(trained_user_set):
                raise ValueError("Speaker Recognition app did not train all requested users")
            for user_id in expected_users:
                accepted = accepted_samples.get(user_id)
                if not isinstance(accepted, int) or accepted < MIN_PROFILE_SAMPLES:
                    raise ValueError(
                        f"Speaker Recognition app accepted too few samples for {user_id}"
                    )
            result = TrainingResult(users_trained=trained_users)
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            _LOGGER.error(
                "Speaker recognition training failed; existing profiles remain usable: %s",
                error,
            )
            return False

        self._enrolled_users.update(result.users_trained)
        self._trained = bool(self.configured_users & self._enrolled_users)
        return True

    async def async_sync_profiles(self) -> bool:
        """Delete backend profiles that are not part of current HA configuration."""
        desired_users = self.configured_users
        try:
            response = await self._async_post(
                "/profiles/sync", {"desired_users": sorted(desired_users)}, timeout_seconds=30
            )
            enrolled = response.get("enrolled_users")
            removed = response.get("removed_users")
            if not isinstance(enrolled, list) or not all(
                isinstance(user, str) for user in enrolled
            ):
                raise ValueError("Invalid profile synchronization response")
            if not isinstance(removed, list) or not all(
                isinstance(user, str) for user in removed
            ):
                raise ValueError("Invalid removed-user list")
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            _LOGGER.error("Unable to synchronize persisted speaker profiles: %s", error)
            return False

        self._enrolled_users = set(enrolled)
        self._trained = bool(desired_users & self._enrolled_users)
        if removed:
            _LOGGER.info("Removed stale speaker profiles: %s", ", ".join(sorted(removed)))
        return True

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        """Recognize speaker from audio data without delaying Assist indefinitely."""
        if not self._trained:
            return None
        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            request_started = perf_counter()
            try:
                response = await asyncio.wait_for(
                    self._async_post(
                        "/recognize",
                        {"audio": {"audio_data": audio_base64, "sample_rate": sample_rate}},
                    ),
                    timeout=RECOGNITION_TIMEOUT_SECONDS,
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
                raw_user_id is not None and not isinstance(raw_user_id, str)
                or not isinstance(candidate_user_id, str)
                or not isinstance(confidence, (int, float))
                or not isinstance(similarity, (int, float))
                or margin is not None and not isinstance(margin, (int, float))
                or not isinstance(accepted, bool)
                or not isinstance(all_scores, dict)
                or not all(
                    isinstance(user, str) and isinstance(score, (int, float))
                    for user, score in all_scores.items()
                )
            ):
                raise ValueError("Invalid recognition response from Speaker Recognition app")

            configured = self.configured_users
            if candidate_user_id not in configured:
                raise ValueError("Backend returned an unconfigured speaker candidate")
            if raw_user_id is not None and raw_user_id not in configured:
                raise ValueError("Backend accepted an unconfigured speaker identity")
            if any(user not in configured for user in all_scores):
                raise ValueError("Backend scored an unconfigured speaker profile")
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
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Speaker recognition exceeded %.1fs; continuing Assist without identity",
                RECOGNITION_TIMEOUT_SECONDS,
            )
            return None
        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during recognition: %s", error)
            return None
        return result

    async def async_denoise(
        self, audio_data: bytes, sample_rate: int
    ) -> DenoiseResult:
        """Request an optional RNNoise diagnostic preview from the backend."""
        response = await self._async_post(
            "/denoise",
            {
                "audio": {
                    "audio_data": base64.b64encode(audio_data).decode("ascii"),
                    "sample_rate": sample_rate,
                }
            },
            timeout_seconds=30,
        )
        encoded = response.get("audio_data")
        returned_rate = response.get("sample_rate")
        processing_seconds = response.get("processing_seconds")
        engine = response.get("engine")
        if (
            not isinstance(encoded, str)
            or not isinstance(returned_rate, int)
            or returned_rate <= 0
            or not isinstance(processing_seconds, (int, float))
            or not isinstance(engine, str)
        ):
            raise ValueError("Invalid denoise response from Speaker Recognition app")
        try:
            denoised = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise ValueError("Invalid denoise audio from Speaker Recognition app") from error
        if not denoised or len(denoised) % 2:
            raise ValueError("Denoise response did not contain valid PCM16 audio")
        return DenoiseResult(
            audio_data=denoised,
            sample_rate=returned_rate,
            processing_seconds=float(processing_seconds),
            engine=engine,
        )

    def update_voice_samples(self, voice_samples: list[dict]) -> None:
        """Update configured voice sample media references."""
        self.voice_samples = voice_samples
