"""Background pairing of authoritative and experimental speaker-engine scores."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, HomeAssistant, callback

from .const import CONF_ENTRY_TYPE, DOMAIN, ENTRY_TYPE_MAIN
from .recognition import SpeakerRecognition
from .telemetry import DecisionHistory

_LOGGER = logging.getLogger(__name__)
_SHADOW_EVENT = "speaker_recognition_shadow_evaluated"


def _main_recognition(hass: HomeAssistant) -> SpeakerRecognition | None:
    """Return the loaded main recognition runtime, if available."""
    entries: list[ConfigEntry] = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) != ENTRY_TYPE_MAIN:
            continue
        runtime = getattr(entry, "runtime_data", None)
        if isinstance(runtime, SpeakerRecognition):
            return runtime
    return None


async def _async_run_shadow(
    hass: HomeAssistant,
    history: DecisionHistory,
    recognition: SpeakerRecognition,
    *,
    pcm_audio: bytes,
    sample_rate: int,
    utterance_sequence: int,
    stt_entity_id: str | None,
) -> None:
    """Collect one non-authoritative result and attach it to the matching turn."""
    result = await recognition.async_shadow_recognize(pcm_audio, sample_rate=sample_rate)
    if result is None:
        return

    payload: dict[str, Any] = {
        "utterance_sequence": utterance_sequence,
        "entity_id": stt_entity_id,
        "engine_id": result.engine_id,
        "candidate_user_id": result.candidate_user_id,
        "similarity": result.similarity,
        "margin": result.margin,
        "all_scores": result.all_scores,
        "processing_seconds": result.processing_seconds,
    }
    if not history.record_shadow_event(payload):
        _LOGGER.debug(
            "Discarded invalid shadow result for utterance %d", utterance_sequence
        )
        return

    hass.bus.async_fire(_SHADOW_EVENT, payload)
    _LOGGER.debug(
        "Recorded %s shadow scores for utterance %d in %.3fs",
        result.engine_id,
        utterance_sequence,
        result.processing_seconds,
    )


def async_setup_shadow_evaluation(
    hass: HomeAssistant, history: DecisionHistory
) -> None:
    """Pair cached Assist PCM with the optional shadow engine in the background."""
    domain_data = hass.data.setdefault(DOMAIN, {})
    if "shadow_evaluation_event_unsub" in domain_data:
        return

    @callback
    def _handle_authoritative_result(event: Event) -> None:
        sequence = event.data.get("utterance_sequence")
        if not isinstance(sequence, int):
            return

        # Live diagnostic turns are deliberately excluded from calibration/history.
        excluded = domain_data.setdefault("calibration_excluded_utterances", set())
        if sequence in excluded:
            return

        cache = domain_data.get("utterance_audio")
        if not isinstance(cache, dict):
            return
        cached = cache.get(sequence)
        if (
            not isinstance(cached, tuple)
            or len(cached) != 2
            or not isinstance(cached[0], bytes)
            or not isinstance(cached[1], int)
        ):
            return
        pcm_audio, sample_rate = cached

        recognition = _main_recognition(hass)
        if recognition is None:
            return

        # Register the authoritative result before the ordinary history listener runs.
        # This preserves backend engine/latency metadata without changing the STT event
        # schema or putting any experimental work on the Assist critical path.
        enriched = dict(event.data)
        diagnostics = recognition.pop_authoritative_diagnostics(pcm_audio)
        if diagnostics is not None:
            enriched["engine_id"] = diagnostics[0]
            enriched["backend_processing_seconds"] = diagnostics[1]
        history.record_event(enriched)

        if not recognition.shadow_ready:
            return
        stt_entity_id = event.data.get("entity_id")
        if not isinstance(stt_entity_id, str):
            stt_entity_id = None
        hass.async_create_task(
            _async_run_shadow(
                hass,
                history,
                recognition,
                pcm_audio=pcm_audio,
                sample_rate=sample_rate,
                utterance_sequence=sequence,
                stt_entity_id=stt_entity_id,
            ),
            f"Speaker Recognition shadow evaluation {sequence}",
        )

    domain_data["shadow_evaluation_event_unsub"] = hass.bus.async_listen(
        "speaker_recognition_detected", _handle_authoritative_result
    )
