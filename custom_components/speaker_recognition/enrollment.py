"""Shared helpers for interactive speaker enrollment."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
import secrets
import sys
import time
import wave

from homeassistant.core import HomeAssistant

from .const import DOMAIN

ENROLLMENT_PHRASES = (
    "The morning light is warm across the kitchen table.",
    "Please turn on the hallway lamp before it gets dark.",
    "My favorite music sounds best on a quiet afternoon.",
    "A small bird landed beside the open garden gate.",
    "Tomorrow I will remember to water all the plants.",
    "Home should feel comfortable, calm, and welcoming.",
)
MIN_ENROLLMENT_SAMPLES = 5
_SESSION_TIMEOUT = 90.0
_COMPLETION_TTL = 120.0
_MEDIA_SUBDIR = "speaker_recognition_enrollment"


@dataclass
class SatelliteEnrollmentSession:
    """One expected satellite utterance for enrollment."""

    session_id: str
    user_id: str
    satellite_id: str
    sample_index: int
    expires_at: float


def _domain_data(hass: HomeAssistant) -> dict:
    return hass.data.setdefault(DOMAIN, {})


def staged_samples(hass: HomeAssistant, user_id: str) -> dict[int, dict[str, str]]:
    """Return staged enrollment samples for a user."""
    staged = _domain_data(hass).setdefault("enrollment_staged", {})
    return staged.setdefault(user_id, {})


def _sample_location(
    hass: HomeAssistant, user_id: str, sample_index: int
) -> tuple[Path, str]:
    """Return an HA media path plus media-source identifier for a sample."""
    if not hass.config.media_dirs:
        raise ValueError("Home Assistant has no local media directory configured")
    media_key, media_root = next(iter(hass.config.media_dirs.items()))
    safe_user = "".join(ch for ch in user_id if ch.isalnum() or ch in "-_")
    relative = f"{_MEDIA_SUBDIR}/{safe_user}/sample_{sample_index + 1}.wav"
    path = Path(media_root) / relative
    media_id = f"media-source://media_source/{media_key}/{relative}"
    return path, media_id


def _write_wav(path: Path, pcm_data: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)


async def async_stage_pcm_sample(
    hass: HomeAssistant,
    user_id: str,
    sample_index: int,
    pcm_data: bytes,
    sample_rate: int,
) -> dict[str, float | bool | str]:
    """Persist one enrollment sample and return basic quality metrics."""
    if not 0 <= sample_index < len(ENROLLMENT_PHRASES):
        raise ValueError("Invalid sample index")
    if sample_rate <= 0 or len(pcm_data) < sample_rate:
        raise ValueError("Enrollment sample must contain at least 0.5 seconds of audio")
    if len(pcm_data) > sample_rate * 2 * 30:
        raise ValueError("Enrollment sample is too long")

    absolute_path, media_id = _sample_location(hass, user_id, sample_index)
    await hass.async_add_executor_job(_write_wav, absolute_path, pcm_data, sample_rate)
    staged_samples(hass, user_id)[sample_index] = {"media_content_id": media_id}

    samples = array("h")
    samples.frombytes(pcm_data[: len(pcm_data) - (len(pcm_data) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    peak = max((abs(value) for value in samples), default=0)
    clipping = sum(abs(value) >= 32700 for value in samples)
    clipping_fraction = clipping / len(samples) if samples else 0.0
    duration = len(pcm_data) / (sample_rate * 2)
    return {
        "media_content_id": media_id,
        "duration": round(duration, 2),
        "peak": round(peak / 32768, 3),
        "too_quiet": peak < 1800,
        "clipping": clipping_fraction > 0.005,
    }


def _satellite_sessions(hass: HomeAssistant) -> dict[str, SatelliteEnrollmentSession]:
    """Return active enrollment sessions indexed by satellite entity ID."""
    return _domain_data(hass).setdefault("enrollment_satellite_sessions", {})


def _completed_satellite_captures(hass: HomeAssistant) -> dict[str, float]:
    """Return recent satellite capture completions and prune stale entries."""
    completed = _domain_data(hass).setdefault("completed_satellite_captures", {})
    cutoff = time.monotonic() - _COMPLETION_TTL
    for session_id, completed_at in list(completed.items()):
        if completed_at < cutoff:
            completed.pop(session_id, None)
    return completed


def completed_satellite_capture_ids(hass: HomeAssistant) -> list[str]:
    """Return IDs of recently completed satellite enrollment captures."""
    return list(_completed_satellite_captures(hass))


def start_satellite_session(
    hass: HomeAssistant, user_id: str, satellite_id: str, sample_index: int
) -> str:
    """Expect exactly one enrollment turn from a chosen satellite."""
    session_id = secrets.token_urlsafe(12)
    _satellite_sessions(hass)[satellite_id] = SatelliteEnrollmentSession(
        session_id=session_id,
        user_id=user_id,
        satellite_id=satellite_id,
        sample_index=sample_index,
        expires_at=time.monotonic() + _SESSION_TIMEOUT,
    )
    return session_id


def cancel_satellite_session(hass: HomeAssistant, satellite_id: str) -> None:
    """Cancel the pending enrollment session for one satellite."""
    _satellite_sessions(hass).pop(satellite_id, None)


async def async_capture_satellite_sample(
    hass: HomeAssistant,
    satellite_id: str | None,
    pcm_data: bytes,
    sample_rate: int,
) -> bool:
    """Capture audio only when it belongs to the explicitly selected satellite."""
    if satellite_id is None:
        return False
    sessions = _satellite_sessions(hass)
    session = sessions.get(satellite_id)
    if not isinstance(session, SatelliteEnrollmentSession):
        return False
    if time.monotonic() > session.expires_at:
        sessions.pop(satellite_id, None)
        return False

    await async_stage_pcm_sample(
        hass,
        session.user_id,
        session.sample_index,
        pcm_data,
        sample_rate,
    )
    sessions.pop(satellite_id, None)
    _completed_satellite_captures(hass)[session.session_id] = time.monotonic()
    return True
