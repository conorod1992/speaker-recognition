"""Frontend panel and WebSocket API for guided microphone enrollment."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from aiohttp import ClientError
import voluptuous as vol

from homeassistant.components import frontend, panel_custom, websocket_api
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .const import (
    CONF_ENTRY_TYPE,
    DOMAIN,
    ENROLLMENT_PHRASES,
    ENROLLMENT_SAMPLE_RATE,
    ENTRY_TYPE_MAIN,
    MAX_ENROLLMENT_SAMPLES,
    MAX_ENROLLMENT_SECONDS,
    MIN_ENROLLMENT_SAMPLES,
    MIN_ENROLLMENT_SECONDS,
)
from .recognition import SpeakerRecognition

PANEL_URL_PATH = "speaker-recognition"
PANEL_STATIC_URL = "/speaker_recognition_frontend"
PANEL_ELEMENT = "speaker-recognition-panel"

_DATA_STATIC_REGISTERED = "enrollment_frontend_static_registered"
_DATA_WS_REGISTERED = "enrollment_frontend_ws_registered"


def _get_main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Return the configured main Speaker Recognition entry."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


def _get_recognition(hass: HomeAssistant) -> SpeakerRecognition | None:
    """Return the active recognition runtime, if available."""
    entry = _get_main_entry(hass)
    if entry is None:
        return None
    runtime = getattr(entry, "runtime_data", None)
    return runtime if isinstance(runtime, SpeakerRecognition) else None


def _decode_pcm_sample(value: str, sample_rate: int) -> bytes:
    """Decode and bound one browser-provided mono 16-bit PCM sample."""
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Enrollment audio is not valid base64 PCM") from error

    if len(data) % 2:
        raise ValueError("Enrollment PCM must contain complete 16-bit samples")

    minimum_bytes = int(sample_rate * 2 * MIN_ENROLLMENT_SECONDS)
    maximum_bytes = int(sample_rate * 2 * MAX_ENROLLMENT_SECONDS)
    if len(data) < minimum_bytes:
        raise ValueError(
            f"Enrollment sample must be at least {MIN_ENROLLMENT_SECONDS:g} seconds"
        )
    if len(data) > maximum_bytes:
        raise ValueError(
            f"Enrollment sample must be at most {MAX_ENROLLMENT_SECONDS:g} seconds"
        )
    return data


async def async_setup_frontend(hass: HomeAssistant) -> None:
    """Register the enrollment panel, static module and WebSocket commands."""
    domain_data = hass.data.setdefault(DOMAIN, {})

    if not domain_data.get(_DATA_STATIC_REGISTERED):
        frontend_dir = Path(__file__).parent / "frontend"
        await hass.http.async_register_static_paths(
            [
                StaticPathConfig(
                    PANEL_STATIC_URL,
                    str(frontend_dir),
                    cache_headers=False,
                )
            ]
        )
        domain_data[_DATA_STATIC_REGISTERED] = True

    if not domain_data.get(_DATA_WS_REGISTERED):
        websocket_api.async_register_command(hass, websocket_enrollment_info)
        websocket_api.async_register_command(hass, websocket_enrollment_train)
        websocket_api.async_register_command(hass, websocket_enrollment_test)
        domain_data[_DATA_WS_REGISTERED] = True

    if not frontend.async_panel_exists(hass, PANEL_URL_PATH):
        await panel_custom.async_register_panel(
            hass=hass,
            frontend_url_path=PANEL_URL_PATH,
            webcomponent_name=PANEL_ELEMENT,
            sidebar_title="Speaker Recognition",
            sidebar_icon="mdi:account-voice",
            module_url=f"{PANEL_STATIC_URL}/panel.js",
            require_admin=True,
        )


@callback
def async_remove_frontend_panel(hass: HomeAssistant) -> None:
    """Remove the panel when the main config entry unloads."""
    if frontend.async_panel_exists(hass, PANEL_URL_PATH):
        frontend.async_remove_panel(hass, PANEL_URL_PATH)


@websocket_api.require_admin
@websocket_api.websocket_command(
    {vol.Required("type"): "speaker_recognition/enrollment/info"}
)
@websocket_api.async_response
async def websocket_enrollment_info(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return enrollment prompts, HA users and current backend profiles."""
    recognition = _get_recognition(hass)
    if recognition is None:
        connection.send_error(
            msg["id"], "not_ready", "Speaker Recognition is not ready"
        )
        return

    users = await hass.auth.async_get_users()
    connection.send_result(
        msg["id"],
        {
            "users": [
                {"id": user.id, "name": user.name or user.id}
                for user in users
                if not user.system_generated and user.is_active
            ],
            "enrolled_users": recognition.enrolled_users,
            "phrases": list(ENROLLMENT_PHRASES),
            "minimum_samples": MIN_ENROLLMENT_SAMPLES,
            "maximum_samples": MAX_ENROLLMENT_SAMPLES,
            "sample_rate": ENROLLMENT_SAMPLE_RATE,
            "minimum_seconds": MIN_ENROLLMENT_SECONDS,
            "maximum_seconds": MAX_ENROLLMENT_SECONDS,
        },
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "speaker_recognition/enrollment/train",
        vol.Required("user_id"): str,
        vol.Required("sample_rate"): vol.In([ENROLLMENT_SAMPLE_RATE]),
        vol.Required("samples"): vol.All(
            [str], vol.Length(min=MIN_ENROLLMENT_SAMPLES, max=MAX_ENROLLMENT_SAMPLES)
        ),
    }
)
@websocket_api.async_response
async def websocket_enrollment_train(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Train a profile from browser-captured PCM without persisting raw audio."""
    recognition = _get_recognition(hass)
    if recognition is None:
        connection.send_error(
            msg["id"], "not_ready", "Speaker Recognition is not ready"
        )
        return

    users = await hass.auth.async_get_users()
    valid_user_ids = {
        user.id for user in users if not user.system_generated and user.is_active
    }
    user_id = msg["user_id"]
    if user_id not in valid_user_ids:
        connection.send_error(msg["id"], "invalid_user", "Unknown Home Assistant user")
        return

    try:
        samples = [
            _decode_pcm_sample(value, msg["sample_rate"]) for value in msg["samples"]
        ]
        result = await recognition.async_train_pcm_samples(
            user_id, samples, msg["sample_rate"]
        )
    except (ClientError, OSError, ValueError, TypeError) as error:
        connection.send_error(msg["id"], "training_failed", str(error))
        return

    connection.send_result(
        msg["id"],
        {
            "trained_users": result.users_trained,
            "profile_consistency": result.profile_consistency.get(user_id),
            "outlier_samples": result.outlier_samples.get(user_id, []),
        },
    )


@websocket_api.require_admin
@websocket_api.websocket_command(
    {
        vol.Required("type"): "speaker_recognition/enrollment/test",
        vol.Required("sample_rate"): vol.In([ENROLLMENT_SAMPLE_RATE]),
        vol.Required("audio_data"): str,
    }
)
@websocket_api.async_response
async def websocket_enrollment_test(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Recognize a fresh microphone sample without storing it."""
    recognition = _get_recognition(hass)
    if recognition is None:
        connection.send_error(
            msg["id"], "not_ready", "Speaker Recognition is not ready"
        )
        return

    try:
        pcm = _decode_pcm_sample(msg["audio_data"], msg["sample_rate"])
    except ValueError as error:
        connection.send_error(msg["id"], "invalid_audio", str(error))
        return

    result = await recognition.async_recognize(pcm, sample_rate=msg["sample_rate"])
    if result is None:
        connection.send_error(
            msg["id"], "recognition_failed", "Speaker recognition produced no result"
        )
        return

    connection.send_result(
        msg["id"],
        {
            "user_id": result.user_id,
            "candidate_user_id": result.candidate_user_id,
            "similarity": result.similarity,
            "margin": result.margin,
            "accepted": result.accepted,
            "all_scores": result.all_scores,
        },
    )
