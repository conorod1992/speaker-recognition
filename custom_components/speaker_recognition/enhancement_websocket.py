"""Home Assistant WebSocket adapter for speech enhancement previews."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant

from .const import CONF_ENTRY_TYPE, DOMAIN, ENTRY_TYPE_MAIN
from .enhancement import (
    build_comparison_preview,
    enhance_speech_pcm,
    wav_base64,
)
from .enhancement_metrics import audio_quality_metrics
from .recognition import SpeakerRecognition


def _main_recognition(hass: HomeAssistant) -> SpeakerRecognition | None:
    """Return the loaded main Speaker Recognition runtime, if available."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ENTRY_TYPE) != ENTRY_TYPE_MAIN:
            continue
        runtime = getattr(entry, "runtime_data", None)
        if isinstance(runtime, SpeakerRecognition):
            return runtime
    return None


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/enhancement_preview",
        vol.Required("utterance_sequence"): int,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_enhancement_preview(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return temporary original/basic/RNNoise comparison audio."""
    sequence = int(msg["utterance_sequence"])
    domain_data = hass.data.setdefault(DOMAIN, {})
    live_result = domain_data.get("live_test_result")
    if (
        not isinstance(live_result, dict)
        or live_result.get("utterance_sequence") != sequence
    ):
        connection.send_error(
            msg["id"],
            "not_live_test_audio",
            "Audio preview is only available for the latest live satellite test",
        )
        return

    audio_cache = domain_data.get("utterance_audio")
    cached = audio_cache.get(sequence) if isinstance(audio_cache, dict) else None
    if (
        not isinstance(cached, tuple)
        or len(cached) != 2
        or not isinstance(cached[0], bytes)
        or not isinstance(cached[1], int)
    ):
        connection.send_error(
            msg["id"], "audio_unavailable", "The live-test audio is no longer cached"
        )
        return

    pcm_data, sample_rate = cached
    basic_started = perf_counter()
    basic_pcm = await hass.async_add_executor_job(
        enhance_speech_pcm, pcm_data, sample_rate
    )
    basic_seconds = perf_counter() - basic_started

    rnnoise_pcm: bytes | None = None
    rnnoise_seconds: float | None = None
    combo_pcm: bytes | None = None
    combo_seconds: float | None = None
    neural_engine: str | None = None
    neural_error: str | None = None
    recognition = _main_recognition(hass)
    if recognition is None:
        neural_error = "Speaker Recognition backend is not currently available"
    else:
        try:
            rnnoise = await recognition.async_denoise(pcm_data, sample_rate)
            combo = await recognition.async_denoise(basic_pcm, sample_rate)
            if rnnoise.sample_rate != sample_rate or combo.sample_rate != sample_rate:
                neural_error = "Neural backend returned an unexpected sample rate"
            else:
                rnnoise_pcm = rnnoise.audio_data
                rnnoise_seconds = rnnoise.processing_seconds
                combo_pcm = combo.audio_data
                combo_seconds = combo.processing_seconds
                neural_engine = rnnoise.engine
        except Exception as error:  # Diagnostics must never disrupt Assist.
            neural_error = f"Neural preview unavailable: {error}"

    result = await hass.async_add_executor_job(
        build_comparison_preview,
        pcm_data,
        basic_pcm,
        sample_rate,
        basic_seconds,
        combo_pcm,
        combo_seconds,
        neural_engine,
        neural_error,
    )
    if rnnoise_pcm is not None:
        result["rnnoise_wav_base64"] = await hass.async_add_executor_job(
            wav_base64, rnnoise_pcm, sample_rate
        )
        result["rnnoise_processing_seconds"] = rnnoise_seconds or 0.0
    result["comparison_metrics"] = {
        "original": await hass.async_add_executor_job(
            audio_quality_metrics, pcm_data, sample_rate
        ),
        "basic": await hass.async_add_executor_job(
            audio_quality_metrics, basic_pcm, sample_rate
        ),
        "rnnoise": await hass.async_add_executor_job(
            audio_quality_metrics, rnnoise_pcm, sample_rate
        )
        if rnnoise_pcm is not None
        else {},
        "combo": await hass.async_add_executor_job(
            audio_quality_metrics, combo_pcm, sample_rate
        )
        if combo_pcm is not None
        else {},
    }
    connection.send_result(msg["id"], result)


def async_register_enhancement_websocket(hass: HomeAssistant) -> None:
    """Register the experimental enhancement preview command."""
    websocket_api.async_register_command(hass, websocket_enhancement_preview)
