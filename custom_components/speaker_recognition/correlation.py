"""Task-local correlation between STT recognition and conversation processing."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrelatedRecognition:
    """Speaker decision produced by one specific Assist STT turn."""

    user_id: str | None
    candidate_user_id: str
    confidence: float
    similarity: float
    margin: float | None
    accepted: bool
    all_scores: dict[str, float]
    stt_entity_id: str | None
    utterance_sequence: int
    stt_seconds: float = 0.0
    recognition_seconds: float = 0.0
    preparation_seconds: float = 0.0
    added_latency_seconds: float = 0.0
    audio_seconds: float | None = None


_current_recognition: ContextVar[CorrelatedRecognition | None] = ContextVar(
    "speaker_recognition_current_turn", default=None
)


def clear_correlated_recognition() -> None:
    """Clear any recognition decision inherited by the current task."""
    _current_recognition.set(None)


def set_correlated_recognition(result: CorrelatedRecognition) -> None:
    """Attach a recognition decision to the current Assist pipeline task."""
    _current_recognition.set(result)


def take_correlated_recognition() -> CorrelatedRecognition | None:
    """Consume the recognition decision attached to the current Assist task."""
    result = _current_recognition.get()
    _current_recognition.set(None)
    return result
