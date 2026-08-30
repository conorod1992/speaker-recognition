"""Shared proxy-source validation and identity helpers."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .const import (
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_STT_ENTITY,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_STT,
)

_PROXY_SOURCE_KEYS = {
    ENTRY_TYPE_STT: CONF_STT_ENTITY,
    ENTRY_TYPE_CONVERSATION: CONF_CONVERSATION_ENTITY,
}
_PROXY_PREFIXES = {
    ENTRY_TYPE_STT: "stt.",
    ENTRY_TYPE_CONVERSATION: "conversation.",
}


def effective_proxy_source(entry: ConfigEntry, entry_type: str) -> str | None:
    """Return the effective source entity for a proxy config entry."""
    key = _PROXY_SOURCE_KEYS.get(entry_type)
    if key is None:
        return None
    value = entry.options.get(key, entry.data.get(key))
    return value if isinstance(value, str) else None


def proxy_unique_id(entry_type: str, source_entity_id: str) -> str:
    """Return the stable config-entry unique ID for a proxy source."""
    return f"{entry_type}_{source_entity_id}"


def proxy_source_in_use(
    hass: HomeAssistant,
    entry_type: str,
    source_entity_id: str,
    *,
    exclude_entry_id: str | None = None,
) -> bool:
    """Return whether another proxy of the same type already wraps this source."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == exclude_entry_id:
            continue
        if entry.data.get(CONF_ENTRY_TYPE) != entry_type:
            continue
        if effective_proxy_source(entry, entry_type) == source_entity_id:
            return True
    return False


def is_speaker_recognition_proxy_entity(
    hass: HomeAssistant, entity_id: str
) -> bool:
    """Return whether an entity belongs to another Speaker Recognition proxy."""
    registry_entry = er.async_get(hass).async_get(entity_id)
    if registry_entry is None or registry_entry.config_entry_id is None:
        return False

    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id != registry_entry.config_entry_id:
            continue
        return entry.data.get(CONF_ENTRY_TYPE) in {
            ENTRY_TYPE_STT,
            ENTRY_TYPE_CONVERSATION,
        }
    return False


def validate_proxy_source(
    hass: HomeAssistant,
    entry_type: str,
    source_entity_id: str,
    *,
    exclude_entry_id: str | None = None,
) -> str | None:
    """Return a config-flow error key when a proxy source is unsafe."""
    expected_prefix = _PROXY_PREFIXES.get(entry_type)
    if expected_prefix is None or not source_entity_id.startswith(expected_prefix):
        return "not_stt_entity" if entry_type == ENTRY_TYPE_STT else "not_conversation_entity"
    if is_speaker_recognition_proxy_entity(hass, source_entity_id):
        return "speaker_recognition_proxy_source"
    if proxy_source_in_use(
        hass,
        entry_type,
        source_entity_id,
        exclude_entry_id=exclude_entry_id,
    ):
        return "proxy_source_already_wrapped"
    return None


def sync_proxy_unique_id(
    hass: HomeAssistant, entry: ConfigEntry, entry_type: str
) -> bool:
    """Synchronize a proxy entry unique ID with its effective wrapped source."""
    source = effective_proxy_source(entry, entry_type)
    if source is None:
        return False
    desired_unique_id = proxy_unique_id(entry_type, source)
    if entry.unique_id == desired_unique_id:
        return True

    for other in hass.config_entries.async_entries(DOMAIN):
        if other.entry_id != entry.entry_id and other.unique_id == desired_unique_id:
            return False

    hass.config_entries.async_update_entry(entry, unique_id=desired_unique_id)
    return True
