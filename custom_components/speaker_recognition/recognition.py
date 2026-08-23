"""Speaker recognition module."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aiohttp import ClientError, ClientTimeout
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .audio import decode_wav

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

DEFAULT_ADDON_URL = "http://localhost:8099"


@dataclass(frozen=True)
class TrainingResult:
    """Training response returned by the Speaker Recognition app."""

    users_trained: list[str]


@dataclass(frozen=True)
class RecognitionResult:
    """Recognition response returned by the Speaker Recognition app."""

    user_id: str
    confidence: float
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
        self._base_url = base_url.rstrip("/")

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

    async def async_train(self) -> None:
        """Train the speaker recognition model with configured voice samples."""
        _LOGGER.debug(
            "Training speaker recognition with %d voice samples",
            len(self.voice_samples),
        )

        if not self.voice_samples:
            _LOGGER.warning("No voice samples configured for training")
            self._trained = False
            return

        try:
            voice_sample_models = []
            for sample in self.voice_samples:
                user_id = sample["user"]
                media_id = sample["samples"].get("media_content_id", "")

                if media_id.startswith("media-source://media_source/local/"):
                    relative_path = media_id.replace(
                        "media-source://media_source/local/", ""
                    )
                    full_path = Path(self.hass.config.path("media")) / relative_path

                    # Read the audio file
                    audio_data = await self.hass.async_add_executor_job(
                        full_path.read_bytes
                    )
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
                else:
                    _LOGGER.warning("Unsupported media_content_id format: %s", media_id)
                    continue

            if not voice_sample_models:
                _LOGGER.warning("No valid training samples prepared")
                self._trained = False
                return

            response = await self._async_post(
                "/train", {"voice_samples": voice_sample_models}
            )
            trained_users = response.get("trained_users")
            if not isinstance(trained_users, list) or not all(
                isinstance(user, str) for user in trained_users
            ):
                raise ValueError("Invalid training response from Speaker Recognition app")
            if not trained_users:
                raise ValueError("Speaker Recognition app did not train any users")
            result = TrainingResult(users_trained=trained_users)

        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.error(
                "Speaker recognition training failed; recognition will be skipped: %s",
                error,
            )
            self._trained = False
        else:
            self._trained = True
            _LOGGER.info(
                "Speaker recognition training completed: %d users trained",
                len(result.users_trained),
            )

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        """Recognize speaker from audio data.

        Args:
            audio_data: Raw audio data to analyze (PCM 16-bit)
            sample_rate: Audio sample rate

        Returns:
            RecognitionResult if a speaker is recognized, None otherwise
        """
        if not self._trained:
            _LOGGER.warning(
                "Speaker recognition is not trained; skipping recognition request"
            )
            return None

        try:
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            response = await self._async_post(
                "/recognize",
                {
                    "audio": {
                        "audio_data": audio_base64,
                        "sample_rate": sample_rate,
                    }
                },
            )
            user_id = response.get("user_id")
            confidence = response.get("confidence")
            all_scores = response.get("all_scores")
            if (
                not isinstance(user_id, str)
                or not isinstance(confidence, (int, float))
                or not isinstance(all_scores, dict)
                or not all(
                    isinstance(user, str) and isinstance(score, (int, float))
                    for user, score in all_scores.items()
                )
            ):
                raise ValueError(
                    "Invalid recognition response from Speaker Recognition app"
                )
            result = RecognitionResult(
                user_id=user_id,
                confidence=float(confidence),
                all_scores={user: float(score) for user, score in all_scores.items()},
            )

        except (ClientError, OSError, ValueError, TypeError) as error:
            _LOGGER.error("Error during recognition: %s", error)
            return None
        else:
            _LOGGER.debug(
                "Recognition result: user=%s, confidence=%.2f",
                result.user_id,
                result.confidence,
            )

            return result

    def update_voice_samples(self, voice_samples: list[dict]) -> None:
        """Update voice samples and mark as needing retraining.

        Args:
            voice_samples: New list of voice samples
        """
        self.voice_samples = voice_samples
        self._trained = False
        _LOGGER.info("Voice samples updated, retraining required")
