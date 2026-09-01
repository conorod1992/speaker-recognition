"""Speaker recognition module."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientResponseError, ClientTimeout
from homeassistant.components import media_source
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .audio import decode_persisted_training_wav, read_bounded_wav

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ADDON_URL = "http://localhost:8099"
RECOGNITION_TIMEOUT_SECONDS = 4.0
SHADOW_RECOGNITION_TIMEOUT_SECONDS = 12.0
MIN_PROFILE_SAMPLES = 3
DEFAULT_ENGINE_ID = "resemblyzer"


class RecognitionBackendUnavailable(RuntimeError):
    """Raised when the Speaker Recognition backend cannot provide valid status."""


@dataclass(frozen=True)
class TrainingResult:
    """Training response returned by the Speaker Recognition app."""

    users_trained: list[str]


@dataclass(frozen=True)
class RecognitionResult:
    """Authoritative recognition response returned by the app."""

    engine_id: str
    user_id: str | None
    candidate_user_id: str
    confidence: float
    similarity: float
    margin: float | None
    accepted: bool
    all_scores: dict[str, float]
    processing_seconds: float = 0.0


@dataclass(frozen=True)
class ShadowRecognitionResult:
    """Non-authoritative raw scores returned by an experimental engine."""

    engine_id: str
    candidate_user_id: str
    similarity: float
    margin: float | None
    all_scores: dict[str, float]
    processing_seconds: float


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
        self._shadow_engine_id: str | None = None
        self._shadow_enrolled_users: set[str] = set()
        self._shadow_pending_users: set[str] = set()
        self._shadow_training_task: asyncio.Task[None] | None = None
        self._authoritative_diagnostics: dict[str, tuple[str, float]] = {}
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
        """Return authoritative backend profile IDs."""
        return set(self._enrolled_users)

    @property
    def shadow_engine_id(self) -> str | None:
        """Return the configured experimental engine, if any."""
        return self._shadow_engine_id

    @property
    def shadow_enrolled_users(self) -> set[str]:
        """Return users with experimental shadow profiles."""
        return set(self._shadow_enrolled_users)

    @property
    def shadow_ready(self) -> bool:
        """Return whether shadow scoring can run without waiting for training."""
        return bool(
            self._shadow_engine_id
            and self.configured_users & self._shadow_enrolled_users
            and (self._shadow_training_task is None or self._shadow_training_task.done())
        )

    @staticmethod
    def _audio_key(audio_data: bytes) -> str:
        """Return a privacy-preserving in-memory key for one PCM utterance."""
        return hashlib.sha256(audio_data).hexdigest()

    def pop_authoritative_diagnostics(
        self, audio_data: bytes
    ) -> tuple[str, float] | None:
        """Consume backend engine/latency metadata for a just-finished STT turn."""
        return self._authoritative_diagnostics.pop(self._audio_key(audio_data), None)

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

    async def _async_training_payload(
        self, user_ids: set[str] | None
    ) -> tuple[list[dict[str, Any]], set[str]]:
        """Build a backend training payload from configured local media."""
        selected_samples = [
            sample
            for sample in self.voice_samples
            if user_ids is None or sample.get("user") in user_ids
        ]
        expected_users = {
            user
            for sample in selected_samples
            if isinstance((user := sample.get("user")), str)
        }
        voice_sample_models: list[dict[str, Any]] = []
        for sample in selected_samples:
            user_id = sample.get("user")
            if not isinstance(user_id, str):
                continue
            selected_media = sample.get("samples", [])
            media_items = (
                selected_media if isinstance(selected_media, list) else [selected_media]
            )
            for media_item in media_items:
                if not isinstance(media_item, dict):
                    continue
                media_id = media_item.get("media_content_id", "")
                if not isinstance(media_id, str) or not media_id.startswith("media-source://"):
                    continue
                audio_data = await self._async_read_media(media_id)
                pcm_data, sample_rate = await self.hass.async_add_executor_job(
                    decode_persisted_training_wav, audio_data
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
        return voice_sample_models, expected_users

    @staticmethod
    def _validate_training_response(
        response: dict[str, Any], expected_users: set[str]
    ) -> list[str]:
        """Validate the shared authoritative/shadow training contract."""
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
        return trained_users

    def _schedule_shadow_training(self, user_ids: set[str]) -> None:
        """Queue best-effort shadow profile training without blocking setup/Assist."""
        if not self._shadow_engine_id or not user_ids:
            return
        self._shadow_pending_users.update(user_ids)
        if self._shadow_training_task is not None and not self._shadow_training_task.done():
            return
        self._shadow_training_task = self.hass.async_create_task(
            self._async_drain_shadow_training(),
            "Speaker Recognition experimental shadow profile training",
        )

    async def _async_drain_shadow_training(self) -> None:
        """Train queued shadow users serially and absorb all experimental failures."""
        try:
            while self._shadow_engine_id and self._shadow_pending_users:
                pending = set(self._shadow_pending_users)
                self._shadow_pending_users.difference_update(pending)
                await self.async_train_shadow(pending)
        except Exception:
            _LOGGER.exception("Unexpected failure in experimental shadow training task")

    def _schedule_shadow_sync(self) -> None:
        """Synchronize shadow removals without delaying authoritative updates."""
        if not self._shadow_engine_id:
            return
        self.hass.async_create_task(
            self.async_sync_shadow_profiles(),
            "Speaker Recognition experimental shadow profile sync",
        )

    async def async_refresh_status(self) -> bool:
        """Refresh authoritative and optional shadow profile status."""
        try:
            response = await self._async_get("/health")
            trained = response.get("trained")
            enrolled_users = response.get("enrolled_users")
            if not isinstance(trained, bool) or not isinstance(enrolled_users, list):
                raise ValueError("Invalid profile status from Speaker Recognition app")
            if not all(isinstance(user, str) and user for user in enrolled_users):
                raise ValueError("Invalid enrolled user list from Speaker Recognition app")

            shadow_engine_id = response.get("shadow_engine_id")
            shadow_users = response.get("shadow_enrolled_users", [])
            shadow_error = response.get("shadow_error")
            if shadow_engine_id is not None and not isinstance(shadow_engine_id, str):
                raise ValueError("Invalid shadow engine ID from Speaker Recognition app")
            if not isinstance(shadow_users, list) or not all(
                isinstance(user, str) and user for user in shadow_users
            ):
                raise ValueError("Invalid shadow enrolled user list")
            if shadow_error is not None and not isinstance(shadow_error, str):
                raise ValueError("Invalid shadow error status")
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            self._trained = False
            self._enrolled_users = set()
            self._shadow_engine_id = None
            self._shadow_enrolled_users = set()
            _LOGGER.warning("Unable to read speaker recognition status: %s", error)
            raise RecognitionBackendUnavailable(
                "Speaker Recognition backend is unavailable"
            ) from error

        self._enrolled_users = set(enrolled_users)
        self._trained = trained and bool(self.configured_users & self._enrolled_users)
        self._shadow_engine_id = shadow_engine_id
        self._shadow_enrolled_users = set(shadow_users)
        _LOGGER.info(
            "Speaker recognition backend has %d persisted authoritative profiles",
            len(self._enrolled_users),
        )

        missing_shadow = self.configured_users - self._shadow_enrolled_users
        if self._shadow_engine_id and missing_shadow and not shadow_error:
            self._schedule_shadow_training(missing_shadow)
        elif self._shadow_engine_id and shadow_error:
            _LOGGER.warning(
                "Experimental shadow engine %s is unavailable: %s",
                self._shadow_engine_id,
                shadow_error,
            )
        return self._trained

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        """Train authoritative profiles, then queue shadow training separately."""
        try:
            voice_sample_models, expected_users = await self._async_training_payload(user_ids)
            if not voice_sample_models:
                _LOGGER.warning("No changed voice samples available for training")
                return False
            response = await self._async_post(
                "/train", {"voice_samples": voice_sample_models}
            )
            trained_users = self._validate_training_response(response, expected_users)
            result = TrainingResult(users_trained=trained_users)
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            _LOGGER.error(
                "Speaker recognition training failed; existing profiles remain usable: %s",
                error,
            )
            return False

        self._enrolled_users.update(result.users_trained)
        self._trained = bool(self.configured_users & self._enrolled_users)
        self._schedule_shadow_training(expected_users)
        return True

    async def async_train_shadow(self, user_ids: set[str]) -> bool:
        """Best-effort training for the experimental engine only."""
        if not self._shadow_engine_id or not user_ids:
            return False
        try:
            voice_sample_models, expected_users = await self._async_training_payload(user_ids)
            if not voice_sample_models:
                return False
            response = await self._async_post(
                "/shadow/train",
                {"voice_samples": voice_sample_models},
                timeout_seconds=300,
            )
            trained_users = self._validate_training_response(response, expected_users)
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            _LOGGER.warning(
                "Experimental shadow profile training failed without affecting "
                "authoritative recognition: %s",
                error,
            )
            return False

        self._shadow_enrolled_users.update(trained_users)
        _LOGGER.info(
            "Experimental %s shadow profiles are ready for %d configured users",
            self._shadow_engine_id,
            len(self.configured_users & self._shadow_enrolled_users),
        )
        return True

    async def async_sync_profiles(self) -> bool:
        """Delete authoritative profiles not part of current HA configuration."""
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
        self._schedule_shadow_sync()
        return True

    async def async_sync_shadow_profiles(self) -> bool:
        """Best-effort cleanup for the experimental profile namespace."""
        if not self._shadow_engine_id:
            return False
        desired_users = self.configured_users
        try:
            response = await self._async_post(
                "/shadow/profiles/sync",
                {"desired_users": sorted(desired_users)},
                timeout_seconds=30,
            )
            enrolled = response.get("enrolled_users")
            if not isinstance(enrolled, list) or not all(
                isinstance(user, str) for user in enrolled
            ):
                raise ValueError("Invalid shadow profile synchronization response")
        except (ClientError, OSError, ValueError, TypeError, asyncio.TimeoutError) as error:
            _LOGGER.warning("Unable to synchronize experimental shadow profiles: %s", error)
            return False
        self._shadow_enrolled_users = set(enrolled)
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
            engine_id = response.get("engine_id", DEFAULT_ENGINE_ID)
            processing_seconds = response.get("processing_seconds", 0.0)
            if (
                raw_user_id is not None and not isinstance(raw_user_id, str)
                or not isinstance(candidate_user_id, str)
                or not isinstance(confidence, (int, float))
                or not isinstance(similarity, (int, float))
                or margin is not None and not isinstance(margin, (int, float))
                or not isinstance(accepted, bool)
                or not isinstance(engine_id, str)
                or not engine_id
                or not isinstance(processing_seconds, (int, float))
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
                engine_id=engine_id,
                user_id=raw_user_id,
                candidate_user_id=candidate_user_id,
                confidence=float(confidence),
                similarity=float(similarity),
                margin=float(margin) if margin is not None else None,
                accepted=accepted,
                all_scores={user: float(score) for user, score in all_scores.items()},
                processing_seconds=float(processing_seconds),
            )
            key = self._audio_key(audio_data)
            self._authoritative_diagnostics[key] = (
                result.engine_id,
                result.processing_seconds,
            )
            while len(self._authoritative_diagnostics) > 16:
                self._authoritative_diagnostics.pop(next(iter(self._authoritative_diagnostics)))
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

    async def async_shadow_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> ShadowRecognitionResult | None:
        """Score with the experimental engine without affecting Assist latency/identity."""
        if not self.shadow_ready:
            return None
        try:
            response = await asyncio.wait_for(
                self._async_post(
                    "/shadow/recognize",
                    {
                        "audio": {
                            "audio_data": base64.b64encode(audio_data).decode("utf-8"),
                            "sample_rate": sample_rate,
                        }
                    },
                    timeout_seconds=SHADOW_RECOGNITION_TIMEOUT_SECONDS,
                ),
                timeout=SHADOW_RECOGNITION_TIMEOUT_SECONDS,
            )
            engine_id = response.get("engine_id")
            candidate = response.get("candidate_user_id")
            similarity = response.get("similarity")
            margin = response.get("margin")
            scores = response.get("all_scores")
            processing_seconds = response.get("processing_seconds")
            if (
                not isinstance(engine_id, str)
                or engine_id != self._shadow_engine_id
                or not isinstance(candidate, str)
                or not isinstance(similarity, (int, float))
                or margin is not None and not isinstance(margin, (int, float))
                or not isinstance(processing_seconds, (int, float))
                or not isinstance(scores, dict)
                or not all(
                    isinstance(user, str) and isinstance(score, (int, float))
                    for user, score in scores.items()
                )
            ):
                raise ValueError("Invalid experimental shadow response")
            configured = self.configured_users
            if candidate not in configured or any(user not in configured for user in scores):
                raise ValueError("Shadow engine returned an unconfigured speaker")
            return ShadowRecognitionResult(
                engine_id=engine_id,
                candidate_user_id=candidate,
                similarity=float(similarity),
                margin=float(margin) if margin is not None else None,
                all_scores={user: float(score) for user, score in scores.items()},
                processing_seconds=float(processing_seconds),
            )
        except asyncio.TimeoutError:
            _LOGGER.debug(
                "Experimental shadow scoring exceeded %.1fs; result discarded",
                SHADOW_RECOGNITION_TIMEOUT_SECONDS,
            )
        except ClientResponseError as error:
            if error.status not in (400, 409):
                _LOGGER.warning("Experimental shadow scoring HTTP failure: %s", error)
        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.warning("Experimental shadow scoring failed: %s", error)
        return None

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
