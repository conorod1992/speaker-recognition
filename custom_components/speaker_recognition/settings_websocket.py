"""WebSocket API for Speaker Recognition frontend settings."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback

from .const import (
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_STT_ENTITY,
    CONF_USE_BASIC_DSP,
    DEFAULT_MIN_CONFIDENCE,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
    effective_backend_url,
    effective_use_basic_dsp,
)


def _entries(hass: HomeAssistant) -> list[ConfigEntry]:
    return hass.config_entries.async_entries(DOMAIN)


def _find_entry(hass: HomeAssistant, entry_id: str) -> ConfigEntry | None:
    return next((entry for entry in _entries(hass) if entry.entry_id == entry_id), None)


def _conversation_threshold(entry: ConfigEntry) -> float:
    value = entry.options.get(
        CONF_MIN_CONFIDENCE,
        entry.data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE),
    )
    return float(value)


@websocket_api.websocket_command({vol.Required("type"): f"{DOMAIN}/settings"})
@websocket_api.require_admin
@callback
def websocket_settings(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return all user-facing configuration already supported by the integration."""
    main: dict[str, Any] | None = None
    stt_entries: list[dict[str, Any]] = []
    conversation_entries: list[dict[str, Any]] = []

    for entry in _entries(hass):
        entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)
        if entry_type == ENTRY_TYPE_MAIN:
            main = {
                "entry_id": entry.entry_id,
                "title": entry.title,
                "backend_url": effective_backend_url(entry.data, entry.options),
            }
        elif entry_type == ENTRY_TYPE_STT:
            stt_entries.append(
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "stt_entity": entry.options.get(
                        CONF_STT_ENTITY, entry.data.get(CONF_STT_ENTITY)
                    ),
                    "use_basic_dsp": effective_use_basic_dsp(
                        entry.data, entry.options
                    ),
                }
            )
        elif entry_type == ENTRY_TYPE_CONVERSATION:
            conversation_entries.append(
                {
                    "entry_id": entry.entry_id,
                    "title": entry.title,
                    "conversation_entity": entry.options.get(
                        CONF_CONVERSATION_ENTITY,
                        entry.data.get(CONF_CONVERSATION_ENTITY),
                    ),
                    "min_confidence": _conversation_threshold(entry),
                }
            )

    connection.send_result(
        msg["id"],
        {
            "main": main,
            "stt_entries": stt_entries,
            "conversation_entries": conversation_entries,
        },
    )


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/update_settings",
        vol.Required("entry_id"): str,
        vol.Optional("backend_url"): str,
        vol.Optional("stt_entity"): str,
        vol.Optional("use_basic_dsp"): bool,
        vol.Optional("conversation_entity"): str,
        vol.Optional("min_confidence"): vol.All(
            vol.Coerce(float), vol.Range(min=0.0, max=1.0)
        ),
    }
)
@websocket_api.require_admin
@callback
def websocket_update_settings(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Persist user-facing settings for one Speaker Recognition config entry."""
    entry = _find_entry(hass, msg["entry_id"])
    if entry is None:
        connection.send_error(msg["id"], "unknown_entry", "Config entry was not found")
        return

    entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)
    options = dict(entry.options)

    if entry_type == ENTRY_TYPE_MAIN:
        backend_url = msg.get("backend_url")
        if not isinstance(backend_url, str) or not backend_url.strip():
            connection.send_error(
                msg["id"], "invalid_backend_url", "Backend URL cannot be empty"
            )
            return
        options[CONF_BACKEND_URL] = backend_url.strip()

    elif entry_type == ENTRY_TYPE_STT:
        stt_entity = msg.get("stt_entity")
        if not isinstance(stt_entity, str) or not stt_entity.startswith("stt."):
            connection.send_error(
                msg["id"], "invalid_stt_entity", "Choose a valid STT entity"
            )
            return
        options[CONF_STT_ENTITY] = stt_entity
        options[CONF_USE_BASIC_DSP] = bool(msg.get("use_basic_dsp", False))

    elif entry_type == ENTRY_TYPE_CONVERSATION:
        conversation_entity = msg.get("conversation_entity")
        if not isinstance(conversation_entity, str) or not conversation_entity.startswith(
            "conversation."
        ):
            connection.send_error(
                msg["id"],
                "invalid_conversation_entity",
                "Choose a valid Conversation entity",
            )
            return
        options[CONF_CONVERSATION_ENTITY] = conversation_entity
        threshold = msg.get("min_confidence")
        if not isinstance(threshold, (int, float)):
            connection.send_error(
                msg["id"], "invalid_threshold", "Confidence threshold is required"
            )
            return
        options[CONF_MIN_CONFIDENCE] = float(threshold)

    else:
        connection.send_error(
            msg["id"], "unsupported_entry", "This config entry cannot be edited here"
        )
        return

    hass.config_entries.async_update_entry(entry, options=options)
    connection.send_result(msg["id"], {"saved": True})


def async_register_settings_websocket(hass: HomeAssistant) -> None:
    """Register frontend settings commands."""
    websocket_api.async_register_command(hass, websocket_settings)
    websocket_api.async_register_command(hass, websocket_update_settings)
