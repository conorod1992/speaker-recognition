"""WebSocket API for interactive speaker enrollment and diagnostics."""

from __future__ import annotations

import base64
from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.components.assist_satellite import AssistSatelliteEntityFeature
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .audio import decode_wav
from .const import (
    CONF_ENTRY_TYPE,
    CONF_SAMPLES,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)
from .enrollment import (
    ENROLLMENT_PHRASES,
    MIN_ENROLLMENT_SAMPLES,
    async_stage_pcm_sample,
    cancel_satellite_session,
    completed_satellite_capture_ids,
    staged_samples,
    start_satellite_session,
)


def _main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


def _replace_user_samples(
    existing: list[dict[str, Any]], user_id: str, samples: list[dict[str, str]]
) -> list[dict[str, Any]]:
    retained = [item for item in existing if item.get(CONF_USER) != user_id]
    retained.append(
        {
            CONF_USER: user_id,
            CONF_SAMPLES: samples,
            "sample_metadata": [
                {"phrase": phrase} for phrase in ENROLLMENT_PHRASES[: len(samples)]
            ],
        }
    )
    return retained


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/status"})
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return enrollment choices and lightweight profile diagnostics."""
    entry = _main_entry(hass)
    configured: list[str] = []
    if entry is not None:
        configured = [
            item[CONF_USER]
            for item in entry.options.get(CONF_VOICE_SAMPLES, [])
            if isinstance(item, dict) and isinstance(item.get(CONF_USER), str)
        ]

    auth_users = await hass.auth.async_get_users()
    users = [
        {"id": user.id, "name": user.name or user.id}
        for user in auth_users
        if not user.system_generated
    ]
    satellites = []
    feature = int(AssistSatelliteEntityFeature.START_CONVERSATION)
    for state in hass.states.async_all("assist_satellite"):
        supported = int(state.attributes.get("supported_features", 0) or 0)
        if supported & feature:
            satellites.append(
                {
                    "entity_id": state.entity_id,
                    "name": state.name,
                    "available": state.state not in ("unavailable", "unknown"),
                }
            )

    staged: dict[str, list[int]] = {}
    domain_data = hass.data.setdefault(DOMAIN, {})
    for user_id, values in domain_data.get("enrollment_staged", {}).items():
        if isinstance(values, dict):
            staged[user_id] = sorted(int(index) for index in values)

    connection.send_result(
        msg["id"],
        {
            "configured": entry is not None,
            "users": users,
            "enrolled_users": configured,
            "phrases": list(ENROLLMENT_PHRASES),
            "minimum_samples": MIN_ENROLLMENT_SAMPLES,
            "satellites": satellites,
            "staged": staged,
            "completed_satellite_captures": completed_satellite_capture_ids(hass),
            "microphone_secure_context_required": True,
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/stage_sample",
        vol.Required("user_id"): str,
        vol.Required("sample_index"): vol.All(int, vol.Range(min=0, max=5)),
        vol.Required("wav_base64"): str,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_stage_sample(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Validate and stage a WAV sample recorded in the browser."""
    try:
        wav_data = base64.b64decode(msg["wav_base64"], validate=True)
        pcm_data, sample_rate = await hass.async_add_executor_job(decode_wav, wav_data)
        quality = await async_stage_pcm_sample(
            hass,
            msg["user_id"],
            msg["sample_index"],
            pcm_data,
            sample_rate,
        )
    except (ValueError, TypeError) as err:
        connection.send_error(msg["id"], "invalid_sample", str(err))
        return
    connection.send_result(msg["id"], quality)


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/start_satellite_sample",
        vol.Required("user_id"): str,
        vol.Required("satellite_id"): str,
        vol.Required("sample_index"): vol.All(int, vol.Range(min=0, max=5)),
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_start_satellite_sample(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Prompt one compatible satellite and bind its next Assist turn to enrollment."""
    satellite_id = msg["satellite_id"]
    state = hass.states.get(satellite_id)
    feature = int(AssistSatelliteEntityFeature.START_CONVERSATION)
    if (
        state is None
        or not satellite_id.startswith("assist_satellite.")
        or not int(state.attributes.get("supported_features", 0) or 0) & feature
    ):
        connection.send_error(
            msg["id"],
            "unsupported_satellite",
            "Satellite cannot start conversations",
        )
        return

    sample_index = msg["sample_index"]
    session_id = start_satellite_session(
        hass, msg["user_id"], satellite_id, sample_index
    )
    phrase = ENROLLMENT_PHRASES[sample_index]
    try:
        await hass.services.async_call(
            "assist_satellite",
            "start_conversation",
            {
                "entity_id": satellite_id,
                "start_message": f"Speaker enrollment. Please say: {phrase}",
                "extra_system_prompt": (
                    "This is a speaker enrollment capture. Do not perform actions from "
                    "the spoken enrollment phrase."
                ),
                "preannounce": True,
            },
            blocking=True,
        )
    except Exception as err:  # Home Assistant may surface platform-specific service errors.
        cancel_satellite_session(hass, satellite_id)
        connection.send_error(msg["id"], "satellite_error", str(err))
        return
    connection.send_result(msg["id"], {"started": True, "session_id": session_id})


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/commit_enrollment",
        vol.Required("user_id"): str,
    }
)
@websocket_api.require_admin
@callback
def websocket_commit_enrollment(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Commit staged samples through the transactional config-entry update path."""
    entry = _main_entry(hass)
    if entry is None:
        connection.send_error(
            msg["id"], "not_configured", "Set up Speaker Recognition first"
        )
        return

    staged = staged_samples(hass, msg["user_id"])
    ordered = [staged[index] for index in sorted(staged)]
    if len(ordered) < MIN_ENROLLMENT_SAMPLES:
        connection.send_error(
            msg["id"],
            "too_few_samples",
            f"At least {MIN_ENROLLMENT_SAMPLES} samples are required",
        )
        return

    options = dict(entry.options)
    options[CONF_VOICE_SAMPLES] = _replace_user_samples(
        list(options.get(CONF_VOICE_SAMPLES, [])), msg["user_id"], ordered
    )
    hass.config_entries.async_update_entry(entry, options=options)
    connection.send_result(msg["id"], {"committed": True, "samples": len(ordered)})


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/test_sample",
        vol.Required("wav_base64"): str,
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_test_sample(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Run an arbitrary browser recording through the current profile decision."""
    entry = _main_entry(hass)
    if entry is None:
        connection.send_error(
            msg["id"], "not_ready", "Speaker Recognition is not ready"
        )
        return
    try:
        wav_data = base64.b64decode(msg["wav_base64"], validate=True)
        pcm_data, sample_rate = await hass.async_add_executor_job(decode_wav, wav_data)
        result = await entry.runtime_data.async_recognize(pcm_data, sample_rate)
    except (ValueError, TypeError) as err:
        connection.send_error(msg["id"], "invalid_sample", str(err))
        return
    if result is None:
        connection.send_result(msg["id"], {"available": False})
        return
    connection.send_result(
        msg["id"],
        {
            "available": True,
            "user_id": result.user_id,
            "candidate_user_id": result.candidate_user_id,
            "similarity": result.similarity,
            "margin": result.margin,
            "accepted": result.accepted,
            "all_scores": result.all_scores,
        },
    )


def async_register_websocket_commands(hass: HomeAssistant) -> None:
    """Register all Speaker Recognition frontend commands once."""
    websocket_api.async_register_command(hass, websocket_status)
    websocket_api.async_register_command(hass, websocket_stage_sample)
    websocket_api.async_register_command(hass, websocket_start_satellite_sample)
    websocket_api.async_register_command(hass, websocket_commit_enrollment)
    websocket_api.async_register_command(hass, websocket_test_sample)
