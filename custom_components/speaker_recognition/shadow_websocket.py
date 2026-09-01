"""WebSocket diagnostics for experimental speaker-engine comparison."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .calibration import analyze_engine_comparison
from .const import CONF_ENTRY_TYPE, DOMAIN, ENTRY_TYPE_MAIN
from .live_evaluation import get_live_model_evaluation
from .live_evaluation_analysis import analyze_live_evaluation
from .recognition import SpeakerRecognition
from .telemetry import get_decision_history


def _shadow_status(hass: HomeAssistant) -> dict[str, Any]:
    """Return live experimental-engine readiness for the comparison UI."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.data.get(CONF_ENTRY_TYPE) != ENTRY_TYPE_MAIN:
            continue
        recognition = getattr(entry, "runtime_data", None)
        if not isinstance(recognition, SpeakerRecognition):
            return {
                "configured": True,
                "enabled": False,
                "engine_id": None,
                "ready": False,
                "configured_users": [],
                "enrolled_users": [],
            }
        return {
            "configured": True,
            "enabled": recognition.shadow_engine_id is not None,
            "engine_id": recognition.shadow_engine_id,
            "ready": recognition.shadow_ready,
            "configured_users": sorted(recognition.configured_users),
            "enrolled_users": sorted(recognition.shadow_enrolled_users),
        }
    return {
        "configured": False,
        "enabled": False,
        "engine_id": None,
        "ready": False,
        "configured_users": [],
        "enrolled_users": [],
    }


def _live_status(hass: HomeAssistant) -> dict[str, Any]:
    evaluation = get_live_model_evaluation(hass)
    records = evaluation.records if evaluation is not None else []
    result = analyze_live_evaluation(records)
    result["shadow_status"] = _shadow_status(hass)
    if evaluation is None:
        result.update(
            {
                "running": False,
                "waiting_for_utterance": False,
                "scoring": False,
                "pending": None,
                "trial_count": 0,
            }
        )
    else:
        result.update(evaluation.status())
    return result


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/shadow_comparison"}
)
@websocket_api.require_admin
@callback
def websocket_shadow_comparison(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Retain the legacy history-based comparison endpoint for compatibility."""
    history = get_decision_history(hass)
    records = history.labelled() if history is not None else []
    result = analyze_engine_comparison(records)
    result["shadow_status"] = _shadow_status(hass)
    connection.send_result(msg["id"], result)


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/evaluation_status"}
)
@websocket_api.require_admin
@callback
def websocket_evaluation_status(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return dedicated live A/B state and explicitly labelled aggregate metrics."""
    connection.send_result(msg["id"], _live_status(hass))


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/evaluation_start"}
)
@websocket_api.require_admin
@callback
def websocket_evaluation_start(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Arm continuous one-at-a-time live evaluation."""
    status = _shadow_status(hass)
    if not status["ready"]:
        connection.send_error(
            msg["id"],
            "shadow_not_ready",
            "The experimental shadow profile is not ready",
        )
        return
    evaluation = get_live_model_evaluation(hass)
    if evaluation is None:
        connection.send_error(
            msg["id"], "evaluation_unavailable", "Live evaluation is unavailable"
        )
        return
    evaluation.start()
    connection.send_result(msg["id"], _live_status(hass))


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/evaluation_stop"}
)
@websocket_api.require_admin
@callback
def websocket_evaluation_stop(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    evaluation = get_live_model_evaluation(hass)
    if evaluation is not None:
        evaluation.stop()
    connection.send_result(msg["id"], _live_status(hass))


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/evaluation_label",
        vol.Required("actual_user_id"): vol.Any(str, None),
    }
)
@websocket_api.require_admin
@callback
def websocket_evaluation_label(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Attach independent ground truth to the pending paired turn."""
    actual_user_id = msg["actual_user_id"]
    configured_users = set(_shadow_status(hass)["configured_users"])
    if actual_user_id is not None and actual_user_id not in configured_users:
        connection.send_error(
            msg["id"], "unknown_user", "Ground truth must be an enrolled speaker or unknown"
        )
        return
    evaluation = get_live_model_evaluation(hass)
    if evaluation is None or evaluation.label_pending(actual_user_id) is None:
        connection.send_error(
            msg["id"], "no_pending_trial", "There is no completed paired trial to label"
        )
        return
    connection.send_result(msg["id"], _live_status(hass))


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/evaluation_discard"}
)
@websocket_api.require_admin
@callback
def websocket_evaluation_discard(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    evaluation = get_live_model_evaluation(hass)
    if evaluation is None or not evaluation.discard_pending():
        connection.send_error(
            msg["id"], "no_pending_trial", "There is no completed trial to discard"
        )
        return
    connection.send_result(msg["id"], _live_status(hass))


@websocket_api.websocket_command(
    {vol.Required("type"): f"{DOMAIN}/evaluation_clear"}
)
@websocket_api.require_admin
@callback
def websocket_evaluation_clear(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Clear only saved A/B results; a pending unlabelled trial is left intact."""
    evaluation = get_live_model_evaluation(hass)
    if evaluation is not None:
        evaluation.clear()
    connection.send_result(msg["id"], _live_status(hass))


def async_register_shadow_websocket(hass: HomeAssistant) -> None:
    """Register experimental comparison commands."""
    websocket_api.async_register_command(hass, websocket_shadow_comparison)
    websocket_api.async_register_command(hass, websocket_evaluation_status)
    websocket_api.async_register_command(hass, websocket_evaluation_start)
    websocket_api.async_register_command(hass, websocket_evaluation_stop)
    websocket_api.async_register_command(hass, websocket_evaluation_label)
    websocket_api.async_register_command(hass, websocket_evaluation_discard)
    websocket_api.async_register_command(hass, websocket_evaluation_clear)
