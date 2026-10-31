"""Shared helpers for interactive speaker enrollment."""

from __future__ import annotations

from array import array
import asyncio
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
    claimed_utterance_sequence: int | None = None


def _domain_data(hass: HomeAssistant) -> dict:
    return hass.data.setdefault(DOMAIN, {})


def _sample_lock(hass: HomeAssistant, user_id: str) -> asyncio.Lock:
    """Serialize writes and discards for one speaker."""
    return _domain_data(hass).setdefault("enrollment_sample_locks", {}).setdefault(
        user_id, asyncio.Lock()
    )


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
    generation = secrets.token_urlsafe(8)
    relative = (
        f"{_MEDIA_SUBDIR}/{safe_user}/"
        f"sample_{sample_index + 1}_{generation}.wav"
    )
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


def _managed_media_path(hass: HomeAssistant, media_id: str) -> Path | None:
    """Resolve only enrollment media created and owned by this integration."""
    for media_key, media_root in hass.config.media_dirs.items():
        prefix = f"media-source://media_source/{media_key}/"
        if not media_id.startswith(prefix):
            continue
        relative = media_id[len(prefix) :]
        if not relative.startswith(f"{_MEDIA_SUBDIR}/"):
            return None
        root = (Path(media_root) / _MEDIA_SUBDIR).resolve()
        candidate = (Path(media_root) / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate
    return None


async def async_discard_staged_samples(hass: HomeAssistant, user_id: str) -> int:
    """Discard uncommitted audio for one user without touching active profiles."""
    async with _sample_lock(hass, user_id):
        staging = _domain_data(hass).setdefault("enrollment_staged", {})
        samples = staging.pop(user_id, {})
        paths = []
        for sample in samples.values():
            if not isinstance(sample, dict):
                continue
            media_id = sample.get("media_content_id")
            if isinstance(media_id, str):
                path = _managed_media_path(hass, media_id)
                if path is not None:
                    paths.append(path)
        if paths:
            await hass.async_add_executor_job(_delete_paths, paths)
        _domain_data(hass).setdefault("enrollment_quality", {}).pop(user_id, None)
        return len(samples)


_ORPHAN_GRACE_SECONDS = 24 * 60 * 60


def _old_unreferenced_recordings(
    roots: list[Path], referenced: set[Path], cutoff: float
) -> list[Path]:
    """Locate old orphaned WAVs only under integration-managed media roots."""
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.glob("*/*.wav"):
            try:
                # A speaker directory itself may be a symlink. Checking only
                # the WAV would allow cleanup to delete files outside this root.
                resolved = path.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    continue
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and resolved not in referenced
                    and path.stat().st_mtime < cutoff
                ):
                    found.append(path)
            except OSError:
                continue
    return found


async def async_cleanup_orphaned_samples(
    hass: HomeAssistant, configured_samples: list[dict]
) -> None:
    """Reclaim interrupted enrollment files after a safe grace period."""
    from .promotions import pending_promotions

    media_ids: set[str] = set()
    for row in configured_samples:
        values = row.get("samples", [])
        for value in values if isinstance(values, list) else [values]:
            if isinstance(value, dict) and isinstance(value.get("media_content_id"), str):
                media_ids.add(value["media_content_id"])
    for staged in _domain_data(hass).get("enrollment_staged", {}).values():
        for value in staged.values():
            if isinstance(value, dict) and isinstance(value.get("media_content_id"), str):
                media_ids.add(value["media_content_id"])
    for item in pending_promotions(hass):
        media_ids.add(item["media_content_id"])
    referenced = set()
    for media_id in media_ids:
        path = _managed_media_path(hass, media_id)
        if path is not None:
            referenced.add(path.resolve())
    roots = [Path(folder).resolve() / _MEDIA_SUBDIR for folder in hass.config.media_dirs.values()]
    stale = await hass.async_add_executor_job(
        _old_unreferenced_recordings, roots, referenced,
        time.time() - _ORPHAN_GRACE_SECONDS,
    )
    if stale:
        await hass.async_add_executor_job(_delete_paths, stale)


def _delete_paths(paths: list[Path]) -> None:
    """Delete integration-owned enrollment files without failing lifecycle updates."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue


async def async_cleanup_managed_samples(
    hass: HomeAssistant, voice_samples: list[dict], user_ids: set[str],
    retained_samples: list[dict] | None = None,
) -> None:
    """Remove superseded panel recordings after a successful profile update."""
    retained_ids = {item.get("media_content_id") for row in retained_samples or []
                    for item in (row.get("samples", []) if isinstance(row.get("samples", []), list) else [row.get("samples")])
                    if isinstance(item, dict)}
    paths: list[Path] = []
    for user_sample in voice_samples:
        if user_sample.get("user") not in user_ids:
            continue
        samples = user_sample.get("samples", [])
        items = samples if isinstance(samples, list) else [samples]
        for item in items:
            if not isinstance(item, dict):
                continue
            media_id = item.get("media_content_id")
            if not isinstance(media_id, str) or media_id in retained_ids:
                continue
            path = _managed_media_path(hass, media_id)
            if path is not None:
                paths.append(path)
    if paths:
        await hass.async_add_executor_job(_delete_paths, paths)


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

    async with _sample_lock(hass, user_id):
        staged = staged_samples(hass, user_id)
        previous = staged.get(sample_index)
        previous_path: Path | None = None
        if isinstance(previous, dict):
            previous_media_id = previous.get("media_content_id")
            if isinstance(previous_media_id, str):
                previous_path = _managed_media_path(hass, previous_media_id)

        absolute_path, media_id = _sample_location(hass, user_id, sample_index)
        await hass.async_add_executor_job(_write_wav, absolute_path, pcm_data, sample_rate)
        staged[sample_index] = {"media_content_id": media_id}

        if previous_path is not None and previous_path != absolute_path:
            await hass.async_add_executor_job(_delete_paths, [previous_path])

    # Imported lazily so capture/media primitives remain independently reusable.
    from .enrollment_quality import schedule_quality_analysis

    schedule_quality_analysis(hass, user_id)
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


_TURN_MARKER_TTL = 10 * 60.0


def _completed_enrollment_turns(hass: HomeAssistant) -> dict[int, float]:
    """Keep time-bounded completion markers until their conversation consumes them."""
    return _domain_data(hass).setdefault("completed_enrollment_turns", {})


def consume_completed_enrollment_turn(hass: HomeAssistant, sequence: int) -> bool:
    """Consume only a recent capture marker from the same Assist turn."""
    turns = _completed_enrollment_turns(hass)
    timestamp = turns.pop(sequence, None)
    return timestamp is not None and time.monotonic() - timestamp <= _TURN_MARKER_TTL


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


def _listening_satellite_ids(hass: HomeAssistant) -> list[str]:
    """Return Assist Satellite entities that are currently listening."""
    return [
        state.entity_id
        for state in hass.states.async_all("assist_satellite")
        if state.state == "listening"
    ]


def claim_satellite_enrollment_turn(
    hass: HomeAssistant, utterance_sequence: int
) -> bool:
    """Claim an armed enrollment only for an unambiguous listening satellite.

    STT metadata does not carry the source satellite. Home Assistant does expose
    satellite listening state before STT starts, so this fallback is safe only when
    exactly one Assist Satellite is listening and that same entity has an armed
    enrollment session. If multiple satellites are listening, the turn is left
    unclaimed rather than risking capture from the wrong room.
    """
    sessions = _satellite_sessions(hass)
    now = time.monotonic()

    for satellite_id, session in list(sessions.items()):
        if now > session.expires_at:
            sessions.pop(satellite_id, None)

    listening = _listening_satellite_ids(hass)
    if len(listening) != 1:
        return False

    session = sessions.get(listening[0])
    if session is None or session.claimed_utterance_sequence is not None:
        return False

    session.claimed_utterance_sequence = utterance_sequence
    return True


async def async_capture_claimed_satellite_sample(
    hass: HomeAssistant,
    utterance_sequence: int,
    pcm_data: bytes,
    sample_rate: int,
) -> bool:
    """Persist audio for the enrollment session claimed by this STT utterance."""
    sessions = _satellite_sessions(hass)
    session = next(
        (
            candidate
            for candidate in sessions.values()
            if candidate.claimed_utterance_sequence == utterance_sequence
        ),
        None,
    )
    if session is None:
        return False
    if time.monotonic() > session.expires_at:
        sessions.pop(session.satellite_id, None)
        return False

    await async_stage_pcm_sample(
        hass,
        session.user_id,
        session.sample_index,
        pcm_data,
        sample_rate,
    )
    sessions.pop(session.satellite_id, None)
    _completed_satellite_captures(hass)[session.session_id] = time.monotonic()
    completed_turns = _completed_enrollment_turns(hass)
    now = time.monotonic()
    # Expire stale entries rather than evicting a still-live earlier turn.
    for sequence, timestamp in list(completed_turns.items()):
        if now - timestamp > _TURN_MARKER_TTL:
            completed_turns.pop(sequence, None)
    completed_turns[utterance_sequence] = now
    _domain_data(hass).setdefault("calibration_excluded_utterances", set()).add(
        utterance_sequence
    )
    return True


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
