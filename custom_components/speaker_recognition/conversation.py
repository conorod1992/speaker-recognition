"""Conversation platform for Speaker Recognition integration."""

from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AbstractConversationAgent,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import (
    Context,
    Event,
    EventStateChangedData,
    HomeAssistant,
    callback,
)
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.intent import IntentResponse, IntentResponseErrorCode

from .const import (
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    DEFAULT_MIN_CONFIDENCE,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)
from .correlation import take_correlated_recognition
from .enrollment import async_capture_satellite_sample
from .recognition import SpeakerRecognition

_LOGGER = logging.getLogger(__name__)


def _get_main_entry(hass: HomeAssistant) -> ConfigEntry | None:
    """Get the main config entry."""
    entries = hass.config_entries.async_entries(DOMAIN)
    for entry in entries:
        if entry.data.get(CONF_ENTRY_TYPE) == ENTRY_TYPE_MAIN:
            return entry
    return None


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Speaker Recognition Conversation platform via config entry."""
    registry = er.async_get(hass)
    conversation_entity_id = config_entry.data[CONF_CONVERSATION_ENTITY]
    entity_id = er.async_validate_entity_id(registry, conversation_entity_id)

    main_entry = _get_main_entry(hass)
    if main_entry is None:
        _LOGGER.error("Main config entry not found")
        return

    async_add_entities(
        [
            SpeakerRecognitionConversationEntity(
                hass,
                config_entry.title,
                entity_id,
                config_entry.entry_id,
                config_entry,
                main_entry,
            )
        ]
    )


class SpeakerRecognitionConversationEntity(
    ConversationEntity, AbstractConversationAgent
):
    """Speaker Recognition Conversation Entity."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry_title: str,
        conversation_entity_id: str,
        unique_id: str,
        config_entry: ConfigEntry,
        main_entry: ConfigEntry,
    ) -> None:
        """Initialize the conversation entity."""
        registry = er.async_get(hass)
        device_registry = dr.async_get(hass)
        wrapped_conversation = registry.async_get(conversation_entity_id)
        device_id = wrapped_conversation.device_id if wrapped_conversation else None
        entity_category = (
            wrapped_conversation.entity_category if wrapped_conversation else None
        )
        has_entity_name = (
            wrapped_conversation.has_entity_name if wrapped_conversation else False
        )

        name: str | None = config_entry_title
        if wrapped_conversation:
            if wrapped_conversation.original_name:
                name = f"{wrapped_conversation.original_name} Speaker Recognition"
            else:
                entity_name = conversation_entity_id.split(".", 1)[-1]
                name = f"{entity_name} Speaker Recognition"

        if device_id and (device := device_registry.async_get(device_id)):
            self.device_entry = device

        self._attr_entity_category = entity_category
        self._attr_has_entity_name = has_entity_name
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._conversation_entity_id = conversation_entity_id
        self._config_entry = config_entry
        self._main_entry = main_entry

        self._cached_languages: list[str] | None | str = None

    @property
    def recognition(self) -> SpeakerRecognition:
        """Get the speaker recognition instance."""
        return self._main_entry.runtime_data

    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold."""
        return self._config_entry.data.get(CONF_MIN_CONFIDENCE, DEFAULT_MIN_CONFIDENCE)

    @callback
    def _async_update_properties(self) -> None:
        """Update cached properties from source entity."""
        source_agent = conversation.async_get_agent(
            self.hass, self._conversation_entity_id
        )
        if source_agent is not None:
            self._cached_languages = source_agent.supported_languages

    @callback
    def _async_state_changed_listener(
        self, event: Event[EventStateChangedData] | None = None
    ) -> None:
        """Handle source entity state changes."""
        if (
            state := self.hass.states.get(self._conversation_entity_id)
        ) is None or state.state == STATE_UNAVAILABLE:
            self._attr_available = False
        else:
            self._attr_available = True
            if self._cached_languages is None:
                self._async_update_properties()

    async def async_added_to_hass(self) -> None:
        """Handle entity added to hass."""
        await super().async_added_to_hass()

        @callback
        def _state_changed_listener(
            event: Event[EventStateChangedData] | None = None,
        ) -> None:
            """Handle child updates."""
            self._async_state_changed_listener(event)
            self.async_write_ha_state()

        self.async_on_remove(
            async_track_state_change_event(
                self.hass, [self._conversation_entity_id], _state_changed_listener
            )
        )
        _state_changed_listener()

    @property
    def supported_languages(self) -> list[str] | str:
        """Return a list of supported languages."""
        return self._cached_languages or []

    async def async_process(self, user_input: ConversationInput) -> ConversationResult:
        """Process a conversation turn."""
        source_agent = conversation.async_get_agent(
            self.hass, self._conversation_entity_id
        )

        if source_agent is None:
            response = IntentResponse(language=user_input.language)
            response.async_set_error(
                IntentResponseErrorCode.FAILED_TO_HANDLE,
                f"Source conversation entity {self._conversation_entity_id} not found",
            )
            return ConversationResult(response=response, conversation_id=None)

        speaker_data = take_correlated_recognition()
        if speaker_data is not None:
            audio_cache = self.hass.data.setdefault(DOMAIN, {}).setdefault(
                "utterance_audio", {}
            )
            cached_audio = audio_cache.pop(speaker_data.utterance_sequence, None)
            if cached_audio is not None:
                pcm_audio, sample_rate = cached_audio
                if await async_capture_satellite_sample(
                    self.hass,
                    user_input.satellite_id,
                    pcm_audio,
                    sample_rate,
                ):
                    response = IntentResponse(language=user_input.language)
                    response.async_set_speech("Voice sample recorded.")
                    return ConversationResult(
                        response=response,
                        conversation_id=user_input.conversation_id,
                    )

            min_confidence = self._config_entry.options.get(
                CONF_MIN_CONFIDENCE,
                self._config_entry.data.get(CONF_MIN_CONFIDENCE, 0.7),
            )

            if (
                speaker_data.accepted
                and speaker_data.confidence >= min_confidence
                and speaker_data.user_id
            ):
                if user_input.context.user_id is None:
                    _LOGGER.info(
                        "Enriching correlated conversation turn: "
                        "recognized_user_id=%s confidence=%.3f utterance=%d "
                        "stt_entity_id=%s device_id=%s satellite_id=%s",
                        speaker_data.user_id,
                        speaker_data.confidence,
                        speaker_data.utterance_sequence,
                        speaker_data.stt_entity_id,
                        user_input.device_id,
                        user_input.satellite_id,
                    )

                    enriched_context = Context(
                        user_id=speaker_data.user_id,
                        parent_id=user_input.context.parent_id,
                        id=user_input.context.id,
                    )
                    user_input = ConversationInput(
                        text=user_input.text,
                        context=enriched_context,
                        conversation_id=user_input.conversation_id,
                        device_id=user_input.device_id,
                        satellite_id=user_input.satellite_id,
                        language=user_input.language,
                        agent_id=user_input.agent_id,
                        extra_system_prompt=user_input.extra_system_prompt,
                    )
                elif user_input.context.user_id != speaker_data.user_id:
                    _LOGGER.debug(
                        "Keeping existing Home Assistant user_id=%s instead of "
                        "correlated speaker recognition user_id=%s",
                        user_input.context.user_id,
                        speaker_data.user_id,
                    )
            else:
                _LOGGER.debug(
                    "Correlated speaker decision not eligible for identity: "
                    "accepted=%s confidence=%.3f threshold=%.3f utterance=%d",
                    speaker_data.accepted,
                    speaker_data.confidence,
                    min_confidence,
                    speaker_data.utterance_sequence,
                )

        return await source_agent.async_process(user_input)
