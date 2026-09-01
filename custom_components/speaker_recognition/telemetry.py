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
_MAX_PENDING_SHADOW = 20
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
    audio_seconds: float | None
    utterance_sequence: int | None
    stt_entity_id: str | None
    engine_id: str = "resemblyzer"
    backend_processing_seconds: float = 0.0
    feedback: str | None = None
    actual_user_id: str | None = None
    shadow_engine_id: str | None = None
    shadow_candidate_user_id: str | None = None
    shadow_similarity: float | None = None
    shadow_margin: float | None = None
    shadow_all_scores: dict[str, float] | None = None
    shadow_processing_seconds: float | None = None


class DecisionHistory:
    """Bounded persistent recognition history."""

    def __init__(self, hass: HomeAssistant) -> None:
        self._store = Store[dict[str, Any]](hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._records: list[dict[str, Any]] = []
        self._pending_shadow: dict[tuple[int, str | None], dict[str, Any]] = {}

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

    @staticmethod
    def _turn_matches(
        item: dict[str, Any], utterance_sequence: int | None, stt_entity_id: str | None
    ) -> bool:
        """Return whether a stored record represents the same STT turn."""
        if utterance_sequence is None:
            return False
        return (
            item.get("utterance_sequence") == utterance_sequence
            and item.get("stt_entity_id") == stt_entity_id
        )

    def _find_turn(
        self, utterance_sequence: int | None, stt_entity_id: str | None
    ) -> dict[str, Any] | None:
        """Find an already-recorded STT turn, newest first."""
        for item in reversed(self._records):
            if self._turn_matches(item, utterance_sequence, stt_entity_id):
                return item
        return None

    @staticmethod
    def _shadow_fields(data: dict[str, Any]) -> dict[str, Any] | None:
        """Validate and normalize one audio-free shadow result."""
        engine_id = data.get("engine_id")
        candidate = data.get("candidate_user_id")
        similarity = data.get("similarity")
        margin = data.get("margin")
        scores = data.get("all_scores")
        processing_seconds = data.get("processing_seconds")
        if (
            not isinstance(engine_id, str)
            or not engine_id
            or not isinstance(candidate, str)
            or not isinstance(similarity, (int, float))
            or margin is not None and not isinstance(margin, (int, float))
            or not isinstance(scores, dict)
            or not isinstance(processing_seconds, (int, float))
        ):
            return None
        return {
            "shadow_engine_id": engine_id,
            "shadow_candidate_user_id": candidate,
            "shadow_similarity": float(similarity),
            "shadow_margin": float(margin) if margin is not None else None,
            "shadow_all_scores": {
                str(user): float(score)
                for user, score in scores.items()
                if isinstance(score, (int, float))
            },
            "shadow_processing_seconds": float(processing_seconds),
        }

    def _apply_pending_shadow(self, item: dict[str, Any]) -> None:
        """Attach a shadow result that happened to finish before the main record."""
        sequence = item.get("utterance_sequence")
        stt_entity_id = item.get("stt_entity_id")
        if not isinstance(sequence, int):
            return
        pending = self._pending_shadow.pop((sequence, stt_entity_id), None)
        if pending is not None:
            item.update(pending)

    def _append(self, record: DecisionRecord) -> str:
        """Append and persist a bounded record."""
        item = asdict(record)
        self._apply_pending_shadow(item)
        self._records.append(item)
        self._records = self._records[-_MAX_DECISIONS:]
        self._schedule_save()
        return record.decision_id

    def record_event(self, data: dict[str, Any]) -> str | None:
        """Record a normal STT recognition event even without a Conversation proxy."""
        sequence = data.get("utterance_sequence")
        stt_entity_id = data.get("entity_id")
        if not isinstance(sequence, int):
            sequence = None
        if not isinstance(stt_entity_id, str):
            stt_entity_id = None

        existing = self._find_turn(sequence, stt_entity_id)
        if existing is not None:
            self._apply_pending_shadow(existing)
            decision_id = existing.get("decision_id")
            return decision_id if isinstance(decision_id, str) else None

        candidate = data.get("candidate_user_id")
        if not isinstance(candidate, str):
            candidate = ""
        user_id = data.get("user_id")
        if not isinstance(user_id, str):
            user_id = None
        margin = data.get("margin")
        if not isinstance(margin, (int, float)):
            margin = None
        all_scores = data.get("all_scores")
        if not isinstance(all_scores, dict):
            all_scores = {}
        engine_id = data.get("engine_id")
        if not isinstance(engine_id, str) or not engine_id:
            engine_id = "resemblyzer"

        record = DecisionRecord(
            decision_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            satellite_id=None,
            user_id=user_id,
            candidate_user_id=candidate,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            similarity=float(data.get("similarity", 0.0) or 0.0),
            margin=float(margin) if margin is not None else None,
            accepted=bool(data.get("accepted")),
            identity_eligible=False,
            threshold=0.0,
            all_scores={
                str(user): float(score)
                for user, score in all_scores.items()
                if isinstance(score, (int, float))
            },
            stt_seconds=float(data.get("stt_seconds", 0.0) or 0.0),
            recognition_seconds=float(data.get("recognition_seconds", 0.0) or 0.0),
            preparation_seconds=float(data.get("preparation_seconds", 0.0) or 0.0),
            added_latency_seconds=float(data.get("added_latency_seconds", 0.0) or 0.0),
            audio_seconds=(
                float(data["audio_seconds"])
                if isinstance(data.get("audio_seconds"), (int, float))
                else None
            ),
            utterance_sequence=sequence,
            stt_entity_id=stt_entity_id,
            engine_id=engine_id,
            backend_processing_seconds=float(
                data.get("backend_processing_seconds", 0.0) or 0.0
            ),
        )
        return self._append(record)

    def record_shadow_event(self, data: dict[str, Any]) -> bool:
        """Enrich one matching decision with non-authoritative shadow scores."""
        sequence = data.get("utterance_sequence")
        stt_entity_id = data.get("entity_id")
        if not isinstance(sequence, int):
            return False
        if not isinstance(stt_entity_id, str):
            stt_entity_id = None
        fields = self._shadow_fields(data)
        if fields is None:
            return False

        existing = self._find_turn(sequence, stt_entity_id)
        if existing is not None:
            existing.update(fields)
            self._schedule_save()
            return True

        self._pending_shadow[(sequence, stt_entity_id)] = fields
        while len(self._pending_shadow) > _MAX_PENDING_SHADOW:
            self._pending_shadow.pop(next(iter(self._pending_shadow)))
        return True

    def record(
        self,
        recognition: CorrelatedRecognition,
        satellite_id: str | None,
        *,
        threshold: float,
        identity_eligible: bool,
    ) -> str:
        """Record or enrich a recognition decision without audio/transcripts."""
        existing = self._find_turn(
            recognition.utterance_sequence, recognition.stt_entity_id
        )
        if existing is not None:
            existing.update(
                {
                    "satellite_id": satellite_id,
                    "user_id": recognition.user_id,
                    "candidate_user_id": recognition.candidate_user_id,
                    "confidence": recognition.confidence,
                    "similarity": recognition.similarity,
                    "margin": recognition.margin,
                    "accepted": recognition.accepted,
                    "identity_eligible": identity_eligible,
                    "threshold": threshold,
                    "all_scores": dict(recognition.all_scores),
                    "engine_id": recognition.engine_id,
                    "backend_processing_seconds": (
                        recognition.backend_processing_seconds
                        if recognition.backend_processing_seconds > 0
                        else float(
                            existing.get("backend_processing_seconds", 0.0) or 0.0
                        )
                    ),
                    "stt_seconds": recognition.stt_seconds,
                    "recognition_seconds": recognition.recognition_seconds,
                    "preparation_seconds": recognition.preparation_seconds,
                    "added_latency_seconds": recognition.added_latency_seconds,
                    "audio_seconds": recognition.audio_seconds,
                }
            )
            self._apply_pending_shadow(existing)
            self._schedule_save()
            decision_id = existing.get("decision_id")
            if isinstance(decision_id, str):
                return decision_id

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
            utterance_sequence=recognition.utterance_sequence,
            stt_entity_id=recognition.stt_entity_id,
            engine_id=recognition.engine_id,
            backend_processing_seconds=recognition.backend_processing_seconds,
        )
        return self._append(record)

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return newest decisions first."""
        return [dict(item) for item in reversed(self._records[-limit:])]

    def labelled(self) -> list[dict[str, Any]]:
        """Return all explicitly labelled decisions in chronological order."""
        return [dict(item) for item in self._records if item.get("feedback")]

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
