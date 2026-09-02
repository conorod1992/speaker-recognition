"""WebSocket helpers for the bounded calibration review queue."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback

from .const import DOMAIN
from .telemetry import get_decision_history

_REVIEW_LIMIT = 10


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/review_decisions"})
@websocket_api.require_admin
@callback
def websocket_review_decisions(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return only the newest ten user-facing calibration decisions."""
    history = get_decision_history(hass)
    if history is None:
        connection.send_result(
            msg["id"], {"decisions": [], "feedback_count": 0, "max_items": _REVIEW_LIMIT}
        )
        return
    decisions = history.review_recent(_REVIEW_LIMIT)
    connection.send_result(
        msg["id"],
        {
            "decisions": decisions,
            "feedback_count": sum(1 for item in decisions if item.get("feedback")),
            "max_items": _REVIEW_LIMIT,
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/decision_audio",
        vol.Required("decision_id"): str,
    }
)
@websocket_api.require_admin
@callback
def websocket_decision_audio(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return one retained mono PCM16 review clip on demand."""
    history = get_decision_history(hass)
    if history is None:
        connection.send_error(msg["id"], "history_unavailable", "History is unavailable")
        return
    clip = history.review_audio_for_decision(msg["decision_id"])
    if clip is None:
        connection.send_error(
            msg["id"],
            "audio_expired",
            "This clip is no longer in the ten-item review queue",
        )
        return
    connection.send_result(msg["id"], clip)


def async_register_review_audio_websocket(hass: HomeAssistant) -> None:
    """Register the bounded review queue and clip playback commands."""
    data = hass.data.setdefault(DOMAIN, {})
    if data.get("review_audio_websocket_registered"):
        return
    websocket_api.async_register_command(hass, websocket_review_decisions)
    websocket_api.async_register_command(hass, websocket_decision_audio)
    data["review_audio_websocket_registered"] = True
