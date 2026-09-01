"""Dedicated live A/B evaluation for speaker embedding engines."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from time import perf_counter
from typing import Any
from uuid import uuid4

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import DOMAIN
from .recognition import (
    RecognitionResult,
    ShadowRecognitionResult,
    SpeakerRecognition,
)

_STORAGE_VERSION = 1
_STORAGE_KEY = f"{DOMAIN}.live_model_evaluation"
_SAVE_DELAY = 1
_CURRENT_TIMEOUT_SECONDS = 30.0


def _domain_data(hass: HomeAssistant) -> dict[str, Any]:
    return hass.data.setdefault(DOMAIN, {})


def _audio_key(audio_data: bytes) -> str:
    return hashlib.sha256(audio_data).hexdigest()


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

    def _prune_stale_current(self) -> None:
        current = self._current
        if not isinstance(current, dict):
            return
        started = current.get("_pair_started_at")
        if isinstance(started, (int, float)) and perf_counter() - started > _CURRENT_TIMEOUT_SECONDS:
            self._current = None

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def clear(self) -> None:
        self._records.clear()
        self._schedule_save()

    def begin_pair(
        self,
        *,
        audio_data: bytes,
        started_at: float,
    ) -> str | None:
        """Arm one live trial at the authoritative model-call start."""
        self._prune_stale_current()
        excluded = _domain_data(self.hass).get("calibration_excluded_utterances")
        if (
            not self.running
            or self.pending is not None
            or self._current is not None
            or isinstance(excluded, set)
            and bool(excluded)
        ):
            return None

        key = _audio_key(audio_data)
        self._current = {
            "trial_id": uuid4().hex,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "_audio_key": key,
            "_pair_started_at": started_at,
        }
        return key

    def _current_for_key(self, key: str) -> dict[str, Any] | None:
        self._prune_stale_current()
        current = self._current
        if not isinstance(current, dict) or current.get("_audio_key") != key:
            return None
        return current

    def record_authoritative_result(
        self,
        key: str,
        result: RecognitionResult | None,
        *,
        call_seconds: float,
    ) -> None:
        current = self._current_for_key(key)
        if current is None:
            return
        if result is None:
            current["authoritative_error"] = "Active engine returned no score"
        else:
            current["authoritative"] = {
                "engine_id": result.engine_id,
                "candidate_user_id": result.candidate_user_id,
                "similarity": result.similarity,
                "margin": result.margin,
                "accepted": result.accepted,
                "user_id": result.user_id,
                "all_scores": dict(result.all_scores),
                "backend_processing_seconds": result.processing_seconds,
                "call_seconds": max(0.0, call_seconds),
            }
        self._maybe_finalize(key)

    def start_shadow_scoring(
        self,
        recognition: SpeakerRecognition,
        *,
        pcm_audio: bytes,
        sample_rate: int,
    ) -> bool:
        """Score ECAPA after the Assist turn, avoiding benchmark CPU interference."""
        key = _audio_key(pcm_audio)
        current = self._current_for_key(key)
        if current is None or current.get("_shadow_started"):
            return False
        current["_shadow_started"] = True

        async def _run() -> None:
            started = perf_counter()
            try:
                result = await recognition.async_shadow_recognize(
                    pcm_audio, sample_rate=sample_rate
                )
            except Exception as error:  # Experimental work must never affect Assist.
                self._record_shadow_failure(key, str(error))
                return
            completed = perf_counter()
            if result is None:
                self._record_shadow_failure(key, "Experimental engine returned no score")
                return
            self._record_shadow_result(
                key,
                result,
                call_seconds=completed - started,
            )

        self.hass.async_create_task(
            _run(), "Speaker Recognition live model evaluation shadow score"
        )
        return True

    def _record_shadow_failure(self, key: str, message: str) -> None:
        current = self._current_for_key(key)
        if current is None:
            return
        current["shadow_error"] = message
        self._maybe_finalize(key)

    def _record_shadow_result(
        self,
        key: str,
        result: ShadowRecognitionResult,
        *,
        call_seconds: float,
    ) -> None:
        current = self._current_for_key(key)
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
        self._maybe_finalize(key)

    def attach_assist_timing(self, pcm_audio: bytes, event_data: dict[str, Any]) -> bool:
        """Bind the paired model calls to the exact Assist turn and STT timing."""
        key = _audio_key(pcm_audio)
        current = self._current_for_key(key)
        if current is None:
            return False
        sequence = event_data.get("utterance_sequence")
        entity_id = event_data.get("entity_id")
        current["utterance_sequence"] = sequence if isinstance(sequence, int) else None
        current["stt_entity_id"] = entity_id if isinstance(entity_id, str) else None
        for name in (
            "stt_seconds",
            "recognition_seconds",
            "preparation_seconds",
            "added_latency_seconds",
            "audio_seconds",
        ):
            value = event_data.get(name)
            current[name] = float(value) if isinstance(value, (int, float)) else None
        current["_timing_attached"] = True
        self._maybe_finalize(key)
        return True

    @staticmethod
    def _apply_effective_latency(current: dict[str, Any], engine: dict[str, Any]) -> None:
        """Estimate model-specific critical-path latency from the real parallel STT turn."""
        call_seconds = engine.get("call_seconds")
        recognition_seconds = current.get("recognition_seconds")
        preparation_seconds = current.get("preparation_seconds")
        pipeline_added = current.get("added_latency_seconds")
        if not all(
            isinstance(value, (int, float))
            for value in (call_seconds, recognition_seconds, preparation_seconds, pipeline_added)
        ):
            return

        # The model call starts after PCM preparation at EOF. When the existing
        # analysis finished after STT, its measured added latency lets us recover
        # the exact STT time that remained at that point. If STT won the race, we
        # only know a lower bound, so any non-zero counterfactual is marked as an
        # upper bound instead of pretending to false precision.
        post_prepare_analysis = max(0.0, recognition_seconds - preparation_seconds)
        if pipeline_added > 0:
            remaining_stt = max(0.0, post_prepare_analysis - pipeline_added)
            engine["effective_added_latency_seconds"] = max(
                0.0, call_seconds - remaining_stt
            )
            engine["effective_added_latency_upper_bound"] = False
            return

        minimum_remaining_stt = post_prepare_analysis
        upper_bound = max(0.0, call_seconds - minimum_remaining_stt)
        engine["effective_added_latency_seconds"] = upper_bound
        engine["effective_added_latency_upper_bound"] = upper_bound > 0

    def _maybe_finalize(self, key: str) -> None:
        current = self._current_for_key(key)
        if current is None or not current.get("_timing_attached"):
            return
        if "authoritative" not in current and "authoritative_error" not in current:
            return
        if "shadow" not in current and "shadow_error" not in current:
            return

        authoritative = current.get("authoritative")
        if isinstance(authoritative, dict):
            self._apply_effective_latency(current, authoritative)
        shadow = current.get("shadow")
        if isinstance(shadow, dict):
            self._apply_effective_latency(current, shadow)

        current.pop("_audio_key", None)
        current.pop("_pair_started_at", None)
        current.pop("_timing_attached", None)
        current.pop("_shadow_started", None)
        self.pending = current
        self._current = None

    def label_pending(self, actual_user_id: str | None) -> dict[str, Any] | None:
        """Persist explicit ground truth for one completed paired turn."""
        if (
            not isinstance(self.pending, dict)
            or "authoritative" not in self.pending
            or "shadow" not in self.pending
        ):
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
        self._prune_stale_current()
        return {
            "running": self.running,
            "waiting_for_utterance": self.running
            and self.pending is None
            and self._current is None,
            "scoring": self._current is not None,
            "pending": dict(self.pending) if isinstance(self.pending, dict) else None,
            "trial_count": len(self._records),
        }


class LiveEvaluationSpeakerRecognition(SpeakerRecognition):
    """SpeakerRecognition runtime that timestamps opt-in A/B trials."""

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        evaluation = get_live_model_evaluation(self.hass)
        started = perf_counter()
        key = (
            evaluation.begin_pair(
                audio_data=audio_data,
                started_at=started,
            )
            if evaluation is not None and self.shadow_ready
            else None
        )
        result = await super().async_recognize(audio_data, sample_rate=sample_rate)
        if key is not None and evaluation is not None:
            evaluation.record_authoritative_result(
                key, result, call_seconds=perf_counter() - started
            )
        return result


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
