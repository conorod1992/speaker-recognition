"""Persist lightweight speaker-recognition decisions for calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .correlation import CorrelatedRecognition

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}.decision_history"
_MAX_DECISIONS = 200
_SAVE_DELAY = 5


@dataclass
class DecisionRecord:
    """One audio-free recognition decision and optional user feedback."""

    decision_id: str
    created_at: str
    satellite_id: str | None
    user_id: str | None
    candidate_user_id: str
    confidence: float
    similarity: float
    margin: float | None
    accepted: bool
    identity_eligible: bool
    threshold: float
    all_scores: dict[str, float]
    stt_seconds: float
    recognition_seconds: float
    preparation_seconds: float
    added_latency_seconds: float
    audio_seconds: float
    feedback: str | None = None
    actual_user_id: str | None = None


class DecisionHistory:
    """Bounded persistent recognition history."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store[dict[str, Any]](hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._records: list[dict[str, Any]] = []

    async def async_load(self) -> None:
        """Load persisted history, tolerating missing or malformed data."""
        data = await self._store.async_load()
        records = data.get("records", []) if isinstance(data, dict) else []
        if isinstance(records, list):
            self._records = [item for item in records if isinstance(item, dict)][
                -_MAX_DECISIONS:
            ]

    def _schedule_save(self) -> None:
        """Coalesce persistence so ordinary Assist use does not write every turn."""
        self._store.async_delay_save(
            lambda: {"records": self._records[-_MAX_DECISIONS:]}, _SAVE_DELAY
        )

    def record(
        self,
        recognition: CorrelatedRecognition,
        satellite_id: str | None,
        *,
        threshold: float,
        identity_eligible: bool,
    ) -> str:
        """Record a recognition decision without retaining speech or transcripts."""
        record = DecisionRecord(
            decision_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            satellite_id=satellite_id,
            user_id=recognition.user_id,
            candidate_user_id=recognition.candidate_user_id,
            confidence=recognition.confidence,
            similarity=recognition.similarity,
            margin=recognition.margin,
            accepted=recognition.accepted,
            identity_eligible=identity_eligible,
            threshold=threshold,
            all_scores=dict(recognition.all_scores),
            stt_seconds=recognition.stt_seconds,
            recognition_seconds=recognition.recognition_seconds,
            preparation_seconds=recognition.preparation_seconds,
            added_latency_seconds=recognition.added_latency_seconds,
            audio_seconds=recognition.audio_seconds,
        )
        self._records.append(asdict(record))
        self._records = self._records[-_MAX_DECISIONS:]
        self._schedule_save()
        return record.decision_id

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return newest decisions first."""
        return [dict(item) for item in reversed(self._records[-limit:])]

    def add_feedback(
        self, decision_id: str, feedback: str, actual_user_id: str | None
    ) -> bool:
        """Attach explicit feedback to one persisted decision."""
        for item in reversed(self._records):
            if item.get("decision_id") == decision_id:
                item["feedback"] = feedback
                item["actual_user_id"] = actual_user_id
                self._schedule_save()
                return True
        return False


def get_decision_history(hass: HomeAssistant) -> DecisionHistory | None:
    """Return the initialized history manager."""
    value = hass.data.get(DOMAIN, {}).get("decision_history")
    return value if isinstance(value, DecisionHistory) else None


async def async_setup_decision_history(hass: HomeAssistant) -> DecisionHistory:
    """Initialize persistent decision history once per HA process."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    existing = domain_data.get("decision_history")
    if isinstance(existing, DecisionHistory):
        return existing

    history = DecisionHistory(hass)
    await history.async_load()
    domain_data["decision_history"] = history
    return history
