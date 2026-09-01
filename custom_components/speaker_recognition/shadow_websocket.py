"""WebSocket diagnostics for experimental speaker-engine comparison."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .calibration import analyze_engine_comparison
from .const import DOMAIN
from .telemetry import get_decision_history


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
    connection.send_result(msg["id"], analyze_engine_comparison(records))


def async_register_shadow_websocket(hass: HomeAssistant) -> None:
    """Register experimental comparison commands."""
    websocket_api.async_register_command(hass, websocket_shadow_comparison)
