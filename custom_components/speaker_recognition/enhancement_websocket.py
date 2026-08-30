"""Home Assistant WebSocket adapter for speech enhancement previews."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant

from .const import DOMAIN
from .enhancement import build_comparison_preview, enhance_speech_pcm
from .enhancement_metrics import audio_quality_metrics


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
    """Return temporary original/basic DSP comparison audio."""
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

    result = await hass.async_add_executor_job(
        build_comparison_preview,
        pcm_data,
        basic_pcm,
        sample_rate,
        basic_seconds,
    )
    result["comparison_metrics"] = {
        "original": await hass.async_add_executor_job(
            audio_quality_metrics, pcm_data, sample_rate
        ),
        "basic": await hass.async_add_executor_job(
            audio_quality_metrics, basic_pcm, sample_rate
        ),
    }
    connection.send_result(msg["id"], result)


def async_register_enhancement_websocket(hass: HomeAssistant) -> None:
    """Register the basic DSP enhancement preview command."""
    websocket_api.async_register_command(hass, websocket_enhancement_preview)
