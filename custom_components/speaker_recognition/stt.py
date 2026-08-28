"""STT platform for Speaker Recognition integration."""

from __future__ import annotations

import asyncio
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
from .correlation import (
    CorrelatedRecognition,
    clear_correlated_recognition,
    set_correlated_recognition,
)
from .recognition import SpeakerRecognition
from .stream import async_process_buffered_stream
from .whisper import WhisperDetection, detect_whisper

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
        """Process audio while keeping recognition bound to this Assist task."""
        clear_correlated_recognition()

        source_entity = async_get_speech_to_text_entity(self.hass, self._stt_entity_id)
        if source_entity is None:
            return SpeechResult(None, SpeechResultState.ERROR)

        domain_data = self.hass.data.setdefault(DOMAIN, {})
        utterance_sequence = int(domain_data.get("utterance_sequence", 0)) + 1
        domain_data["utterance_sequence"] = utterance_sequence

        stt_seconds = 0.0
        recognition_seconds = 0.0
        preparation_seconds = 0.0
        audio_seconds: float | None = None
        stt_completed_at: float | None = None
        recognition_completed_at: float | None = None

        async def process_stt(buffered_stream: AsyncIterable[bytes]) -> SpeechResult:
            nonlocal stt_seconds, stt_completed_at
            stt_started = perf_counter()
            try:
                return await source_entity.async_process_audio_stream(
                    metadata, buffered_stream
                )
            finally:
                stt_completed_at = perf_counter()
                stt_seconds = stt_completed_at - stt_started
                _LOGGER.debug(
                    "Wrapped STT completed in %.3fs for utterance %d",
                    stt_seconds,
                    utterance_sequence,
                )

        async def recognize_speaker(audio_data: bytes):
            nonlocal recognition_seconds, preparation_seconds
            nonlocal audio_seconds, recognition_completed_at
            recognition_started = perf_counter()
            try:
                if (
                    metadata.format != AudioFormats.WAV
                    or metadata.codec != AudioCodecs.PCM
                    or metadata.bit_rate != AudioBitRates.BITRATE_16
                ):
                    _LOGGER.warning(
                        "Skipping speaker and whisper analysis for unsupported STT audio: "
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
                preparation_seconds = perf_counter() - preparation_started
                audio_seconds = len(pcm_audio) / (sample_rate * 2)
                _LOGGER.debug(
                    "Prepared recognition audio in %.3fs for utterance %d",
                    preparation_seconds,
                    utterance_sequence,
                )
                audio_cache = domain_data.setdefault("utterance_audio", {})
                audio_cache[utterance_sequence] = (pcm_audio, sample_rate)
                for old_sequence in sorted(audio_cache)[:-8]:
                    audio_cache.pop(old_sequence, None)

                async def analyze_whisper() -> WhisperDetection:
                    try:
                        return await self.hass.async_add_executor_job(
                            detect_whisper,
                            pcm_audio,
                            sample_rate,
                        )
                    except Exception:  # Supplemental analysis must never block Assist.
                        _LOGGER.exception(
                            "Whisper detection failed for utterance %d",
                            utterance_sequence,
                        )
                        return WhisperDetection(False, 0.0, False)

                recognition_result, whisper_result = await asyncio.gather(
                    self.recognition.async_recognize(
                        pcm_audio,
                        sample_rate=sample_rate,
                    ),
                    analyze_whisper(),
                )
                return recognition_result, whisper_result
            finally:
                recognition_completed_at = perf_counter()
                recognition_seconds = recognition_completed_at - recognition_started
                _LOGGER.debug(
                    "Total speaker and whisper analysis took %.3fs for utterance %d",
                    recognition_seconds,
                    utterance_sequence,
                )

        result, analysis_result = await async_process_buffered_stream(
            stream, process_stt, recognize_speaker
        )

        added_latency_seconds = 0.0
        if stt_completed_at is not None and recognition_completed_at is not None:
            added_latency_seconds = max(0.0, recognition_completed_at - stt_completed_at)

        recognition_result = None
        whisper_result = WhisperDetection(False, 0.0, False)
        if analysis_result is not None:
            recognition_result, whisper_result = analysis_result

        if recognition_result is None:
            _LOGGER.debug(
                "Speaker recognition produced no result for utterance %d",
                utterance_sequence,
            )
            correlated = CorrelatedRecognition(
                user_id=None,
                candidate_user_id="",
                confidence=0.0,
                similarity=0.0,
                margin=None,
                accepted=False,
                all_scores={},
                stt_entity_id=self.entity_id,
                utterance_sequence=utterance_sequence,
                whispering=whisper_result.whispering,
                whisper_score=whisper_result.score,
                whisper_available=whisper_result.available,
                stt_seconds=stt_seconds,
                recognition_seconds=recognition_seconds,
                preparation_seconds=preparation_seconds,
                added_latency_seconds=added_latency_seconds,
                audio_seconds=audio_seconds,
            )
        else:
            correlated = CorrelatedRecognition(
                user_id=recognition_result.user_id,
                candidate_user_id=recognition_result.candidate_user_id,
                confidence=recognition_result.confidence,
                similarity=recognition_result.similarity,
                margin=recognition_result.margin,
                accepted=recognition_result.accepted,
                all_scores=recognition_result.all_scores,
                stt_entity_id=self.entity_id,
                utterance_sequence=utterance_sequence,
                whispering=whisper_result.whispering,
                whisper_score=whisper_result.score,
                whisper_available=whisper_result.available,
                stt_seconds=stt_seconds,
                recognition_seconds=recognition_seconds,
                preparation_seconds=preparation_seconds,
                added_latency_seconds=added_latency_seconds,
                audio_seconds=audio_seconds,
            )

            _LOGGER.info(
                "Speaker recognition decision - User: %s, Candidate: %s, "
                "Similarity: %.3f, Margin: %s, Accepted: %s, "
                "Whispering: %s (%.2f), Recognition: %.3fs, "
                "Added Assist latency: %.3fs, All scores: %s",
                recognition_result.user_id,
                recognition_result.candidate_user_id,
                recognition_result.similarity,
                (
                    f"{recognition_result.margin:.3f}"
                    if recognition_result.margin is not None
                    else "n/a"
                ),
                recognition_result.accepted,
                whisper_result.whispering,
                whisper_result.score,
                recognition_seconds,
                added_latency_seconds,
                {
                    user: f"{score:.3f}"
                    for user, score in recognition_result.all_scores.items()
                },
            )

        set_correlated_recognition(correlated)
        self.hass.bus.async_fire(
            "speaker_recognition_detected",
            {
                "user_id": correlated.user_id,
                "candidate_user_id": correlated.candidate_user_id,
                "confidence": correlated.confidence,
                "similarity": correlated.similarity,
                "margin": correlated.margin,
                "accepted": correlated.accepted,
                "all_scores": correlated.all_scores,
                "whispering": correlated.whispering,
                "whisper_score": correlated.whisper_score,
                "whisper_available": correlated.whisper_available,
                "entity_id": self.entity_id,
                "utterance_sequence": utterance_sequence,
                "stt_seconds": stt_seconds,
                "recognition_seconds": recognition_seconds,
                "preparation_seconds": preparation_seconds,
                "added_latency_seconds": added_latency_seconds,
                "audio_seconds": audio_seconds,
            },
        )

        return result