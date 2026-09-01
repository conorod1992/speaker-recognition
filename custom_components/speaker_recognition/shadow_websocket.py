"""WebSocket diagnostics for experimental speaker-engine comparison."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .calibration import analyze_engine_comparison
from .const import CONF_ENTRY_TYPE, DOMAIN, ENTRY_TYPE_MAIN
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
    """Compare paired labelled authoritative/shadow decisions."""
    history = get_decision_history(hass)
    records = history.labelled() if history is not None else []
    result = analyze_engine_comparison(records)
    result["shadow_status"] = _shadow_status(hass)
    connection.send_result(msg["id"], result)


def async_register_shadow_websocket(hass: HomeAssistant) -> None:
    """Register experimental comparison commands."""
    websocket_api.async_register_command(hass, websocket_shadow_comparison)
