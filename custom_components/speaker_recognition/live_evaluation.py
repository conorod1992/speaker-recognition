"""Dedicated live A/B evaluation for speaker embedding engines."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .correlation import CorrelatedRecognition
from .recognition import ShadowRecognitionResult, SpeakerRecognition

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}.live_model_evaluation"
_SAVE_DELAY = 1


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    return hass.data.setdefault(DOMAIN, {})


class LiveModelEvaluation:
    """Collect explicitly labelled paired engine trials until the user clears them."""

    def __init__(self, hass: HomeAssistant) -> None:
        self.hass = hass
        self._store = Store[dict[str, Any]](hass, _STORAGE_VERSION, _STORAGE_KEY)
        self._records: list[dict[str, Any]] = []
        self.running = False
        self.pending: dict[str, Any] | None = None
        self._current: dict[str, Any] | None = None

    async def async_load(self) -> None:
        data = await self._store.async_load()
        records = data.get("records", []) if isinstance(data, dict) else []
        if isinstance(records, list):
            self._records = [dict(item) for item in records if isinstance(item, dict)]

    def _schedule_save(self) -> None:
        self._store.async_delay_save(lambda: {"records": self._records}, _SAVE_DELAY)

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._records]

    @property
    def current_sequence(self) -> int | None:
        if not isinstance(self._current, dict):
            return None
        value = self._current.get("utterance_sequence")
        return value if isinstance(value, int) else None

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def clear(self) -> None:
        self._records.clear()
        self._schedule_save()

    def claim_turn(self, utterance_sequence: int, stt_entity_id: str | None) -> bool:
        """Claim the next ordinary Assist turn while a live evaluation is running."""
        if not self.running or self.pending is not None or self._current is not None:
            return False
        self._current = {
            "trial_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "utterance_sequence": utterance_sequence,
            "stt_entity_id": stt_entity_id,
        }
        return True

    def _current_for(self, utterance_sequence: int) -> dict[str, Any] | None:
        if not isinstance(self._current, dict):
            return None
        if self._current.get("utterance_sequence") != utterance_sequence:
            return None
        return self._current

    def start_shadow_scoring(
        self,
        recognition: SpeakerRecognition,
        *,
        pcm_audio: bytes,
        sample_rate: int,
        utterance_sequence: int,
    ) -> None:
        """Start ECAPA beside the active recognizer without awaiting it in Assist."""
        if self._current_for(utterance_sequence) is None:
            return

        async def _run() -> None:
            started = perf_counter()
            try:
                result = await recognition.async_shadow_recognize(
                    pcm_audio, sample_rate=sample_rate
                )
            except Exception as error:  # Experimental scoring must never affect Assist.
                self._record_shadow_failure(utterance_sequence, str(error))
                return
            completed = perf_counter()
            if result is None:
                self._record_shadow_failure(
                    utterance_sequence, "Experimental engine returned no score"
                )
                return
            self._record_shadow_result(
                utterance_sequence,
                result,
                call_seconds=completed - started,
                completed_at=completed,
            )

        self.hass.async_create_task(
            _run(), f"Speaker Recognition live model evaluation {utterance_sequence}"
        )

    def _record_shadow_failure(self, utterance_sequence: int, message: str) -> None:
        current = self._current_for(utterance_sequence)
        if current is None:
            return
        current["shadow_error"] = message
        self._maybe_finalize(utterance_sequence)

    def _record_shadow_result(
        self,
        utterance_sequence: int,
        result: ShadowRecognitionResult,
        *,
        call_seconds: float,
        completed_at: float,
    ) -> None:
        current = self._current_for(utterance_sequence)
        if current is None:
            return
        current["shadow"] = {
            "engine_id": result.engine_id,
            "candidate_user_id": result.candidate_user_id,
            "similarity": result.similarity,
            "margin": result.margin,
            "all_scores": dict(result.all_scores),
            "backend_processing_seconds": result.processing_seconds,
            "call_seconds": max(0.0, call_seconds),
        }
        current["_shadow_completed_at"] = completed_at
        self._maybe_finalize(utterance_sequence)

    def record_authoritative(
        self,
        recognition: CorrelatedRecognition,
        *,
        model_call_seconds: float | None,
        model_completed_at: float | None,
        stt_completed_at: float | None,
    ) -> None:
        """Attach the active-engine result and timing anchors for the claimed turn."""
        current = self._current_for(recognition.utterance_sequence)
        if current is None:
            return
        current["stt_seconds"] = recognition.stt_seconds
        current["authoritative"] = {
            "engine_id": recognition.engine_id,
            "candidate_user_id": recognition.candidate_user_id,
            "similarity": recognition.similarity,
            "margin": recognition.margin,
            "accepted": recognition.accepted,
            "user_id": recognition.user_id,
            "all_scores": dict(recognition.all_scores),
            "backend_processing_seconds": recognition.backend_processing_seconds,
            "call_seconds": (
                max(0.0, model_call_seconds)
                if isinstance(model_call_seconds, (int, float))
                else None
            ),
        }
        current["_stt_completed_at"] = stt_completed_at
        current["_authoritative_completed_at"] = model_completed_at
        self._maybe_finalize(recognition.utterance_sequence)

    def _maybe_finalize(self, utterance_sequence: int) -> None:
        current = self._current_for(utterance_sequence)
        if current is None or "authoritative" not in current:
            return
        if "shadow" not in current and "shadow_error" not in current:
            return

        stt_completed = current.pop("_stt_completed_at", None)
        authoritative_completed = current.pop("_authoritative_completed_at", None)
        shadow_completed = current.pop("_shadow_completed_at", None)
        if isinstance(stt_completed, (int, float)):
            authoritative = current.get("authoritative")
            if isinstance(authoritative, dict) and isinstance(
                authoritative_completed, (int, float)
            ):
                authoritative["effective_added_latency_seconds"] = max(
                    0.0, authoritative_completed - stt_completed
                )
            shadow = current.get("shadow")
            if isinstance(shadow, dict) and isinstance(shadow_completed, (int, float)):
                shadow["effective_added_latency_seconds"] = max(
                    0.0, shadow_completed - stt_completed
                )

        self.pending = current
        self._current = None

    def label_pending(self, actual_user_id: str | None) -> dict[str, Any] | None:
        """Persist explicit ground truth for the pending paired turn."""
        if not isinstance(self.pending, dict) or "shadow" not in self.pending:
            return None
        record = dict(self.pending)
        record["actual_user_id"] = actual_user_id
        self._records.append(record)
        self.pending = None
        self._schedule_save()
        return dict(record)

    def discard_pending(self) -> bool:
        if self.pending is None:
            return False
        self.pending = None
        return True

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "waiting_for_utterance": self.running
            and self.pending is None
            and self._current is None,
            "scoring": self._current is not None,
            "pending": dict(self.pending) if isinstance(self.pending, dict) else None,
            "trial_count": len(self._records),
        }


async def async_setup_live_model_evaluation(hass: HomeAssistant) -> LiveModelEvaluation:
    data = _domain_data(hass)
    existing = data.get("live_model_evaluation")
    if isinstance(existing, LiveModelEvaluation):
        return existing
    evaluation = LiveModelEvaluation(hass)
    await evaluation.async_load()
    data["live_model_evaluation"] = evaluation
    return evaluation


def get_live_model_evaluation(hass: HomeAssistant) -> LiveModelEvaluation | None:
    value = _domain_data(hass).get("live_model_evaluation")
    return value if isinstance(value, LiveModelEvaluation) else None


def claim_live_model_evaluation_turn(
    hass: HomeAssistant, utterance_sequence: int, stt_entity_id: str | None
) -> bool:
    evaluation = get_live_model_evaluation(hass)
    return bool(
        evaluation
        and evaluation.claim_turn(utterance_sequence, stt_entity_id)
    )


def is_live_model_evaluation_turn(hass: HomeAssistant, utterance_sequence: int) -> bool:
    evaluation = get_live_model_evaluation(hass)
    return bool(evaluation and evaluation.current_sequence == utterance_sequence)
