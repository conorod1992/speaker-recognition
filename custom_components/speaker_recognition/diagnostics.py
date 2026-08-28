"""Live speaker-recognition diagnostics for real Assist satellite turns."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any
from uuid import uuid4

from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .correlation import CorrelatedRecognition

_LIVE_TEST_TIMEOUT = 90.0


@dataclass(frozen=True)
class LiveTestSession:
    """One expected normal Assist turn from a selected satellite."""

    session_id: str
    satellite_id: str
    expires_at: float
    claimed_utterance_sequence: int | None = None


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    """Return integration runtime data."""
    return hass.data.setdefault(DOMAIN, {})


def start_live_test(hass: HomeAssistant, satellite_id: str) -> str:
    """Arm a one-turn live recognition test for a selected satellite."""
    session_id = uuid4().hex
    data = _domain_data(hass)
    data["live_test_session"] = LiveTestSession(
        session_id=session_id,
        satellite_id=satellite_id,
        expires_at=time.monotonic() + _LIVE_TEST_TIMEOUT,
    )
    data.pop("live_test_result", None)
    return session_id


def live_test_status(hass: HomeAssistant) -> tuple[LiveTestSession | None, dict[str, Any] | None]:
    """Return the active live-test session and latest result, pruning expiry."""
    data = _domain_data(hass)
    session = data.get("live_test_session")
    if isinstance(session, LiveTestSession) and time.monotonic() > session.expires_at:
        data.pop("live_test_session", None)
        session = None
    if not isinstance(session, LiveTestSession):
        session = None

    result = data.get("live_test_result")
    return session, result if isinstance(result, dict) else None


def claim_live_test_turn(hass: HomeAssistant, utterance_sequence: int) -> bool:
    """Claim the armed live test when its selected satellite enters STT."""
    data = _domain_data(hass)
    session = data.get("live_test_session")
    if not isinstance(session, LiveTestSession):
        return False
    if time.monotonic() > session.expires_at:
        data.pop("live_test_session", None)
        return False
    if session.claimed_utterance_sequence is not None:
        return False

    state = hass.states.get(session.satellite_id)
    if state is None or state.state != "listening":
        return False

    data["live_test_session"] = replace(
        session, claimed_utterance_sequence=utterance_sequence
    )
    return True


def record_live_test_result(
    hass: HomeAssistant,
    satellite_id: str | None,
    recognition: CorrelatedRecognition,
    *,
    threshold: float,
    identity_eligible: bool,
) -> bool:
    """Store one correlated decision when it matches the armed live test."""
    data = _domain_data(hass)
    session = data.get("live_test_session")
    if not isinstance(session, LiveTestSession):
        return False
    if time.monotonic() > session.expires_at:
        data.pop("live_test_session", None)
        return False

    claimed_match = (
        session.claimed_utterance_sequence is not None
        and session.claimed_utterance_sequence == recognition.utterance_sequence
    )
    satellite_match = satellite_id == session.satellite_id
    if not claimed_match and not satellite_match:
        return False

    data["live_test_result"] = {
        "session_id": session.session_id,
        "satellite_id": session.satellite_id,
        "user_id": recognition.user_id,
        "candidate_user_id": recognition.candidate_user_id,
        "confidence": recognition.confidence,
        "similarity": recognition.similarity,
        "margin": recognition.margin,
        "accepted": recognition.accepted,
        "all_scores": recognition.all_scores,
        "threshold": threshold,
        "identity_eligible": identity_eligible,
        "whispering": recognition.whispering,
        "whisper_score": recognition.whisper_score,
        "whisper_available": recognition.whisper_available,
        "stt_seconds": recognition.stt_seconds,
        "recognition_seconds": recognition.recognition_seconds,
        "preparation_seconds": recognition.preparation_seconds,
        "added_latency_seconds": recognition.added_latency_seconds,
        "audio_seconds": recognition.audio_seconds,
        "utterance_sequence": recognition.utterance_sequence,
        "stt_entity_id": recognition.stt_entity_id,
    }
    data.pop("live_test_session", None)
    return True
