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
from .calibration import analyze_thresholds
from .const import (
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_SAMPLES,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DEFAULT_MIN_CONFIDENCE,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
)
from .diagnostics import live_test_status, start_live_test
from .enrollment import (
    ENROLLMENT_PHRASES,
    MIN_ENROLLMENT_SAMPLES,
    async_stage_pcm_sample,
    cancel_satellite_session,
    completed_satellite_capture_ids,
    staged_samples,
    start_satellite_session,
)
from .telemetry import get_decision_history

_FEEDBACK_VALUES = ("correct", "wrong_speaker", "missed_speaker")


def _main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


def _conversation_entries(hass: HomeAssistant) -> list[ConfigEntry]:
    """Return configured Speaker Recognition Conversation proxy entries."""
    return [
        entry
        for entry in hass.config_entries.async_entries(DOMAIN)
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_CONVERSATION
    ]


def _conversation_threshold(entry: ConfigEntry) -> float:
    """Return the effective HA confidence threshold for one proxy entry."""
    value = entry.options.get(
        CONF_MIN_CONFIDENCE,
        entry.data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
    )
    return float(value)


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
    enrollment_satellites = []
    start_conversation_feature = int(AssistSatelliteEntityFeature.START_CONVERSATION)
    for state in hass.states.async_all("assist_satellite"):
        satellite = {
            "entity_id": state.entity_id,
            "name": state.name,
            "available": state.state not in ("unavailable", "unknown"),
        }
        satellites.append(satellite)
        supported = int(state.attributes.get("supported_features", 0) or 0)
        if supported & start_conversation_feature:
            enrollment_satellites.append(satellite)

    staged: dict[str, list[int]] = {}
    domain_data = hass.data.setdefault(DOMAIN, {})
    for user_id, values in domain_data.get("enrollment_staged", {}).items():
        if isinstance(values, dict):
            staged[user_id] = sorted(int(index) for index in values)

    live_session, live_result = live_test_status(hass)
    connection.send_result(
        msg["id"],
        {
            "configured": entry is not None,
            "users": users,
            "enrolled_users": configured,
            "phrases": list(ENROLLMENT_PHRASES),
            "minimum_samples": MIN_ENROLLMENT_SAMPLES,
            "satellites": satellites,
            "enrollment_satellites": enrollment_satellites,
            "staged": staged,
            "completed_satellite_captures": completed_satellite_capture_ids(hass),
            "live_test_active": (
                {
                    "session_id": live_session.session_id,
                    "satellite_id": live_session.satellite_id,
                }
                if live_session is not None
                else None
            ),
            "live_test_result": live_result,
            "microphone_secure_context_required": True,
        },
    )


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/decision_history"})
@websocket_api.require_admin
@callback
def websocket_decision_history(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return recent audio-free decisions for calibration."""
    history = get_decision_history(hass)
    if history is None:
        connection.send_result(msg["id"], {"decisions": []})
        return
    decisions = history.recent(25)
    connection.send_result(
        msg["id"],
        {
            "decisions": decisions,
            "feedback_count": sum(1 for item in decisions if item.get("feedback")),
        },
    )


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/calibration_analysis"})
@websocket_api.require_admin
@callback
def websocket_calibration_analysis(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return evidence-based threshold guidance for each Conversation proxy."""
    history = get_decision_history(hass)
    records = history.labelled() if history is not None else []
    entries = []
    for entry in _conversation_entries(hass):
        threshold = _conversation_threshold(entry)
        entries.append(
            {
                "entry_id": entry.entry_id,
                "title": entry.title,
                "conversation_entity": entry.options.get(
                    CONF_CONVERSATION_ENTITY,
                    entry.data.get(CONF_CONVERSATION_ENTITY),
                ),
                "analysis": analyze_thresholds(records, threshold),
            }
        )
    connection.send_result(msg["id"], {"conversation_entries": entries})


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/apply_recommended_threshold",
        vol.Required("entry_id"): str,
    }
)
@websocket_api.require_admin
@callback
def websocket_apply_recommended_threshold(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Apply the current server-side recommendation to one Conversation proxy."""
    entry = next(
        (
            candidate
            for candidate in _conversation_entries(hass)
            if candidate.entry_id == msg["entry_id"]
        ),
        None,
    )
    if entry is None:
        connection.send_error(
            msg["id"], "unknown_conversation_entry", "Conversation proxy was not found"
        )
        return

    history = get_decision_history(hass)
    if history is None:
        connection.send_error(msg["id"], "history_unavailable", "History is unavailable")
        return

    current_threshold = _conversation_threshold(entry)
    analysis = analyze_thresholds(history.labelled(), current_threshold)
    recommendation = analysis.get("recommended_threshold")
    if not analysis.get("ready") or not isinstance(recommendation, (int, float)):
        connection.send_error(
            msg["id"],
            "insufficient_evidence",
            "More labelled recognition decisions are required before applying a recommendation",
        )
        return

    options = dict(entry.options)
    options[CONF_MIN_CONFIDENCE] = float(recommendation)
    hass.config_entries.async_update_entry(entry, options=options)
    connection.send_result(
        msg["id"],
        {
            "applied": True,
            "previous_threshold": current_threshold,
            "new_threshold": float(recommendation),
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/decision_feedback",
        vol.Required("decision_id"): str,
        vol.Required("feedback"): vol.In(_FEEDBACK_VALUES),
        vol.Optional("actual_user_id"): vol.Any(str, None),
    }
)
@websocket_api.require_admin
@websocket_api.async_response
async def websocket_decision_feedback(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Attach explicit ground-truth feedback to a recent decision."""
    history = get_decision_history(hass)
    if history is None:
        connection.send_error(msg["id"], "history_unavailable", "History is unavailable")
        return

    feedback = msg["feedback"]
    actual_user_id = msg.get("actual_user_id")
    if feedback in ("wrong_speaker", "missed_speaker"):
        if not isinstance(actual_user_id, str) or not actual_user_id:
            connection.send_error(
                msg["id"], "actual_user_required", "Choose the actual speaker"
            )
            return
        auth_users = await hass.auth.async_get_users()
        valid_users = {user.id for user in auth_users if not user.system_generated}
        if actual_user_id not in valid_users:
            connection.send_error(
                msg["id"], "unknown_user", "The selected Home Assistant user was not found"
            )
            return
    else:
        actual_user_id = None

    if not history.add_feedback(msg["decision_id"], feedback, actual_user_id):
        connection.send_error(msg["id"], "unknown_decision", "Decision was not found")
        return
    connection.send_result(msg["id"], {"saved": True})


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
        vol.Required("type"): f"{DOMAIN}/start_live_test",
        vol.Required("satellite_id"): str,
    }
)
@websocket_api.require_admin
@callback
def websocket_start_live_test(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Arm one normal Assist turn from a selected satellite for diagnostics."""
    satellite_id = msg["satellite_id"]
    state = hass.states.get(satellite_id)
    if state is None or not satellite_id.startswith("assist_satellite."):
        connection.send_error(
            msg["id"], "unknown_satellite", "Assist satellite was not found"
        )
        return
    if state.state in ("unavailable", "unknown"):
        connection.send_error(
            msg["id"], "satellite_unavailable", "Assist satellite is unavailable"
        )
        return

    session_id = start_live_test(hass, satellite_id)
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
    websocket_api.async_register_command(hass, websocket_decision_history)
    websocket_api.async_register_command(hass, websocket_calibration_analysis)
    websocket_api.async_register_command(hass, websocket_apply_recommended_threshold)
    websocket_api.async_register_command(hass, websocket_decision_feedback)
    websocket_api.async_register_command(hass, websocket_stage_sample)
    websocket_api.async_register_command(hass, websocket_start_satellite_sample)
    websocket_api.async_register_command(hass, websocket_start_live_test)
    websocket_api.async_register_command(hass, websocket_commit_enrollment)
    websocket_api.async_register_command(hass, websocket_test_sample)
