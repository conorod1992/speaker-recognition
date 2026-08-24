"""STT platform for Speaker Recognition integration."""

from __future__ import annotations

from collections.abc import AsyncIterable
import logging
from time import perf_counter

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
    async_get_speech_to_text_entity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import STATE_UNAVAILABLE
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, callback
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_track_state_change_event

from .audio import prepare_live_pcm
from .const import CONF_ENTRY_TYPE, CONF_STT_ENTITY, DOMAIN, ENTRY_TYPE_MAIN
from .recognition import SpeakerRecognition
from .stream import async_process_buffered_stream

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
    """Set up Speaker Recognition STT platform via config entry."""
    registry = er.async_get(hass)
    stt_entity_id = config_entry.data[CONF_STT_ENTITY]
    entity_id = er.async_validate_entity_id(registry, stt_entity_id)

    main_entry = _get_main_entry(hass)
    if main_entry is None:
        _LOGGER.error("Main config entry not found")
        return

    async_add_entities(
        [
            SpeakerRecognitionSTTEntity(
                hass,
                config_entry.title,
                entity_id,
                config_entry.entry_id,
                main_entry,
            )
        ]
    )


class SpeakerRecognitionSTTEntity(SpeechToTextEntity):
    """Speaker Recognition STT Entity."""

    _attr_should_poll = False

    def __init__(
        self,
        hass: HomeAssistant,
        config_entry_title: str,
        stt_entity_id: str,
        unique_id: str,
        main_entry: ConfigEntry,
    ) -> None:
        """Initialize the STT entity."""
        registry = er.async_get(hass)
        device_registry = dr.async_get(hass)
        wrapped_stt = registry.async_get(stt_entity_id)
        device_id = wrapped_stt.device_id if wrapped_stt else None
        entity_category = wrapped_stt.entity_category if wrapped_stt else None
        has_entity_name = wrapped_stt.has_entity_name if wrapped_stt else False

        name: str | None = config_entry_title
        if wrapped_stt:
            if wrapped_stt.original_name:
                name = f"{wrapped_stt.original_name} Speaker Recognition"
            else:
                entity_name = stt_entity_id.split(".", 1)[-1]
                name = f"{entity_name} Speaker Recognition"

        if device_id and (device := device_registry.async_get(device_id)):
            self.device_entry = device

        self._attr_entity_category = entity_category
        self._attr_has_entity_name = has_entity_name
        self._attr_name = name
        self._attr_unique_id = unique_id
        self._stt_entity_id = stt_entity_id
        self._main_entry = main_entry

        self._cached_languages: list[str] | None = None
        self._cached_formats: list[AudioFormats] | None = None
        self._cached_codecs: list[AudioCodecs] | None = None
        self._cached_bit_rates: list[AudioBitRates] | None = None
        self._cached_sample_rates: list[AudioSampleRates] | None = None
        self._cached_channels: list[AudioChannels] | None = None

    @callback
    def _async_update_properties(self) -> None:
        """Update cached properties from source entity."""
        source_entity = async_get_speech_to_text_entity(self.hass, self._stt_entity_id)
        if source_entity is not None:
            self._cached_languages = source_entity.supported_languages
            self._cached_formats = source_entity.supported_formats
            self._cached_codecs = source_entity.supported_codecs
            self._cached_bit_rates = source_entity.supported_bit_rates
            self._cached_sample_rates = source_entity.supported_sample_rates
            self._cached_channels = source_entity.supported_channels

    @callback
    def _async_state_changed_listener(
        self, event: Event[EventStateChangedData] | None = None
    ) -> None:
        """Handle source entity state changes."""
        if (
            state := self.hass.states.get(self._stt_entity_id)
        ) is None or state.state == STATE_UNAVAILABLE:
            self._attr_available = False
        else:
            self._attr_available = True
            # Update cached properties if not yet set
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
                self.hass, [self._stt_entity_id], _state_changed_listener
            )
        )

        # Call once on adding to initialize
        _state_changed_listener()

    @property
    def recognition(self) -> SpeakerRecognition:
        """Get the speaker recognition instance."""
        return self._main_entry.runtime_data

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return self._cached_languages or []

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return self._cached_formats or []

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return self._cached_codecs or []

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bit rates."""
        return self._cached_bit_rates or []

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported sample rates."""
        return self._cached_sample_rates or []

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return self._cached_channels or []

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service.

        This collects audio, performs speaker recognition, and forwards to the source STT.
        """
        # Get the source entity - it should be available if we're being called
        source_entity = async_get_speech_to_text_entity(self.hass, self._stt_entity_id)

        if source_entity is None:
            # Entity not found - return error
            return SpeechResult(None, SpeechResultState.ERROR)

        domain_data = self.hass.data.setdefault(DOMAIN, {})
        utterance_sequence = int(domain_data.get("utterance_sequence", 0)) + 1
        domain_data["utterance_sequence"] = utterance_sequence

        async def process_stt(buffered_stream: AsyncIterable[bytes]) -> SpeechResult:
            stt_started = perf_counter()
            try:
                return await source_entity.async_process_audio_stream(
                    metadata, buffered_stream
                )
            finally:
                _LOGGER.debug(
                    "Wrapped STT completed in %.3fs for utterance %d",
                    perf_counter() - stt_started,
                    utterance_sequence,
                )

        async def recognize_speaker(audio_data: bytes):
            recognition_started = perf_counter()
            try:
                if (
                    metadata.format != AudioFormats.WAV
                    or metadata.codec != AudioCodecs.PCM
                    or metadata.bit_rate != AudioBitRates.BITRATE_16
                ):
                    _LOGGER.warning(
                        "Skipping speaker recognition for unsupported STT audio: "
                        "format=%s codec=%s bit_rate=%s",
                        metadata.format,
                        metadata.codec,
                        metadata.bit_rate,
                    )
                    return None

                preparation_started = perf_counter()
                pcm_audio, sample_rate = await self.hass.async_add_executor_job(
                    prepare_live_pcm,
                    audio_data,
                    int(metadata.sample_rate),
                    int(metadata.channel),
                )
                _LOGGER.debug(
                    "Prepared recognition audio in %.3fs for utterance %d",
                    perf_counter() - preparation_started,
                    utterance_sequence,
                )
                return await self.recognition.async_recognize(
                    pcm_audio, sample_rate=sample_rate
                )
            finally:
                _LOGGER.debug(
                    "Total speaker recognition took %.3fs for utterance %d",
                    perf_counter() - recognition_started,
                    utterance_sequence,
                )

        result, recognition_result = await async_process_buffered_stream(
            stream, process_stt, recognize_speaker
        )

        if recognition_result is None:
            _LOGGER.debug(
                "Speaker recognition produced no result for utterance %d",
                utterance_sequence,
            )
            return result

        _LOGGER.info(
            "Speaker recognition result - User: %s, Confidence: %.3f, All scores: %s",
            recognition_result.user_id,
            recognition_result.confidence,
            {
                user: f"{score:.3f}"
                for user, score in recognition_result.all_scores.items()
            },
        )
        self.hass.bus.async_fire(
            "speaker_recognition_detected",
            {
                "user_id": recognition_result.user_id,
                "confidence": recognition_result.confidence,
                "all_scores": recognition_result.all_scores,
                "entity_id": self.entity_id,
                "utterance_sequence": utterance_sequence,
            },
        )

        # Concurrent utterances may finish out of order. Only the newest-started
        # completed utterance is eligible to become the shared conversation result.
        last_sequence = int(domain_data.get("last_result_sequence", 0))
        if utterance_sequence >= last_sequence:
            domain_data["last_result_sequence"] = utterance_sequence
            domain_data["last_result"] = {
                "user_id": recognition_result.user_id,
                "confidence": recognition_result.confidence,
                "timestamp": self.hass.loop.time(),
                "utterance_sequence": utterance_sequence,
            }

        return result
