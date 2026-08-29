"""The Speaker Recognition integration."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import Event, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers.typing import ConfigType

from .const import (
    CONF_ENTRY_TYPE,
    CONF_PENDING_ENROLLMENT,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
    effective_backend_url,
)
from .enhancement_websocket import async_register_enhancement_websocket
from .frontend import async_register_frontend
from .lifecycle import (
    EnrollmentUpdateFailed,
    async_apply_enrollment_update,
    async_initialize_recognition,
)
from .recognition import RecognitionBackendUnavailable, SpeakerRecognition
from .telemetry import async_setup_decision_history
from .websocket import async_register_websocket_commands

SpeakerRecognitionConfigEntry = ConfigEntry[SpeakerRecognition]


def _get_main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the main config entry."""
    entries = hass.config_entries.async_entries(__name__.rsplit(".", maxsplit=1)[-1])
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up shared frontend, calibration storage and WebSocket resources."""
    await async_register_frontend(hass)
    history = await async_setup_decision_history(hass)
    domain_data = hass.data.setdefault(DOMAIN, {})
    if "decision_history_event_unsub" not in domain_data:

        @callback
        def _record_stt_decision(event: Event) -> None:
            """Persist STT recognition even when no Conversation proxy is used."""
            sequence = event.data.get("utterance_sequence")
            excluded = domain_data.setdefault("calibration_excluded_utterances", set())
            if isinstance(sequence, int) and sequence in excluded:
                excluded.discard(sequence)
                return
            history.record_event(dict(event.data))

        domain_data["decision_history_event_unsub"] = hass.bus.async_listen(
            "speaker_recognition_detected", _record_stt_decision
        )
    async_register_websocket_commands(hass)
    async_register_enhancement_websocket(hass)
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Speaker Recognition from a config entry."""
    entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

    if entry_type == ENTRY_TYPE_MAIN:
        return await async_setup_main_entry(hass, entry)
    if entry_type == ENTRY_TYPE_STT:
        return await async_setup_stt_entry(hass, entry)
    return await async_setup_conversation_entry(hass, entry)


async def async_setup_main_entry(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> bool:
    """Set up main config entry."""
    backend_url = effective_backend_url(entry.data, entry.options)
    voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])

    recognition = SpeakerRecognition(hass, voice_samples, backend_url)

    pending_user_value = entry.options.get(CONF_PENDING_ENROLLMENT)
    pending_user = pending_user_value if isinstance(pending_user_value, str) else None
    try:
        pending_succeeded = await async_initialize_recognition(
            recognition, pending_user
        )
    except RecognitionBackendUnavailable as error:
        raise ConfigEntryNotReady(
            "Speaker Recognition backend is not ready"
        ) from error

    if CONF_PENDING_ENROLLMENT in entry.options and pending_succeeded:
        updated_options = dict(entry.options)
        updated_options.pop(CONF_PENDING_ENROLLMENT)
        hass.config_entries.async_update_entry(entry, options=updated_options)

    entry.runtime_data = recognition
    entry.async_on_unload(entry.add_update_listener(async_update_main_listener))

    return True


async def async_setup_stt_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up STT proxy entry."""
    main_entry = _get_main_entry(hass)
    if main_entry is None:
        return False

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.STT])
    entry.async_on_unload(entry.add_update_listener(async_update_stt_listener))

    return True


async def async_setup_conversation_entry(
    hass: HomeAssistant, entry: ConfigEntry
) -> bool:
    """Set up Conversation proxy entry."""
    main_entry = _get_main_entry(hass)
    if main_entry is None:
        return False

    await hass.config_entries.async_forward_entry_setups(entry, [Platform.CONVERSATION])
    entry.async_on_unload(entry.add_update_listener(async_update_conversation_listener))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    entry_type = entry.data.get(CONF_ENTRY_TYPE, ENTRY_TYPE_MAIN)

    if entry_type == ENTRY_TYPE_MAIN:
        return True

    platforms = (
        [Platform.STT] if entry_type == ENTRY_TYPE_STT else [Platform.CONVERSATION]
    )
    return await hass.config_entries.async_unload_platforms(entry, platforms)


async def async_update_main_listener(
    hass: HomeAssistant, entry: SpeakerRecognitionConfigEntry
) -> None:
    """Handle main config options update."""
    voice_samples = entry.options.get(CONF_VOICE_SAMPLES, [])
    try:
        changed_users = await async_apply_enrollment_update(
            entry.runtime_data, voice_samples
        )
    except EnrollmentUpdateFailed as error:
        updated_options = dict(entry.options)
        updated_options[CONF_VOICE_SAMPLES] = error.previous_samples
        hass.config_entries.async_update_entry(entry, options=updated_options)
        return

    staged = hass.data.get(DOMAIN, {}).get("enrollment_staged")
    if changed_users and isinstance(staged, dict):
        for user_id in changed_users:
            staged.pop(user_id, None)

    await hass.config_entries.async_reload(entry.entry_id)


async def async_update_stt_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle STT proxy options update."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_update_conversation_listener(
    hass: HomeAssistant, entry: ConfigEntry
) -> None:
    """Handle Conversation proxy options update."""
    await hass.config_entries.async_reload(entry.entry_id)
