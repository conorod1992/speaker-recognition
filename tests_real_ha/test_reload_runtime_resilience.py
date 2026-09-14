"""Real Home Assistant acceptance tests for reload and runtime resilience."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from unittest.mock import patch

import pytest
from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)
from custom_components.speaker_recognition.correlation import (
    clear_correlated_recognition,
    take_correlated_recognition,
)
from custom_components.speaker_recognition.recognition import RecognitionResult
from custom_components.speaker_recognition.stt import SpeakerRecognitionSTTEntity


SOURCE = "stt.reload_source"


def _main_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: "",
        },
        options={CONF_VOICE_SAMPLES: []},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


def _entity(hass: HomeAssistant, main: MockConfigEntry) -> SpeakerRecognitionSTTEntity:
    entity = SpeakerRecognitionSTTEntity(
        hass,
        "Reload resilience proxy",
        SOURCE,
        "reload-resilience-proxy",
        main,
    )
    entity.hass = hass
    return entity


def _metadata() -> stt.SpeechMetadata:
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
        bit_rate=stt.AudioBitRates.BITRATE_16,
    )


async def _stream(marker: int = 1) -> AsyncIterable[bytes]:
    yield bytes([marker, 0]) * 4000
    await asyncio.sleep(0)
    yield bytes([marker, 0]) * 4000


class BlockingSource:
    """STT source that exposes the reload window before EOF."""

    supported_languages = ["en-US"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    def __init__(self) -> None:
        self.first_chunk_seen = asyncio.Event()
        self.continue_reading = asyncio.Event()

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        del metadata
        first = True
        async for _chunk in stream:
            if first:
                first = False
                self.first_chunk_seen.set()
                await self.continue_reading.wait()
        return stt.SpeechResult("ok", stt.SpeechResultState.SUCCESS)


class IdentityRecognition:
    """Deterministic recognition runtime representing one config-entry generation."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.calls = 0

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult:
        del audio_data, sample_rate
        self.calls += 1
        return RecognitionResult(
            engine_id="resemblyzer",
            user_id=self.user_id,
            candidate_user_id=self.user_id,
            confidence=0.95,
            similarity=0.90,
            margin=0.30,
            accepted=True,
            all_scores={self.user_id: 0.90},
        )

    def pop_authoritative_diagnostics(self, audio_data: bytes):
        del audio_data
        return None


class TimeoutRecognition:
    """Recognition runtime that simulates a backend timeout."""

    async def async_recognize(self, audio_data: bytes, sample_rate: int = 16000):
        del audio_data, sample_rate
        raise TimeoutError("backend timed out")

    def pop_authoritative_diagnostics(self, audio_data: bytes):
        del audio_data
        return None


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_inflight_stt_turn_keeps_runtime_generation_when_main_runtime_is_replaced(
    hass: HomeAssistant,
) -> None:
    """A main-entry reload cannot switch recognizers halfway through one utterance."""
    old_user = await hass.auth.async_create_user("Old generation")
    new_user = await hass.auth.async_create_user("New generation")
    main = _main_entry(hass)
    old_runtime = IdentityRecognition(old_user.id)
    new_runtime = IdentityRecognition(new_user.id)
    main.runtime_data = old_runtime
    entity = _entity(hass, main)
    source = BlockingSource()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        task = asyncio.create_task(
            entity.async_process_audio_stream(_metadata(), _stream())
        )
        await source.first_chunk_seen.wait()
        main.runtime_data = new_runtime
        source.continue_reading.set()
        result = await task

    correlated = take_correlated_recognition()
    assert result.result is stt.SpeechResultState.SUCCESS
    assert correlated is not None
    assert correlated.user_id == old_user.id
    assert old_runtime.calls == 1
    assert new_runtime.calls == 0


@pytest.mark.asyncio
async def test_backend_timeout_is_supplemental_and_stt_still_succeeds(
    hass: HomeAssistant,
) -> None:
    """A recognition timeout cannot invalidate the wrapped STT result."""
    main = _main_entry(hass)
    main.runtime_data = TimeoutRecognition()
    entity = _entity(hass, main)
    source = BlockingSource()
    source.continue_reading.set()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        result = await entity.async_process_audio_stream(_metadata(), _stream())

    correlated = take_correlated_recognition()
    assert result.result is stt.SpeechResultState.SUCCESS
    assert result.text == "ok"
    assert correlated is not None
    assert correlated.user_id is None
    assert correlated.accepted is False


@pytest.mark.asyncio
async def test_missing_source_after_previous_success_cannot_reuse_stale_identity(
    hass: HomeAssistant,
) -> None:
    """A disabled or removed source fails closed on the next turn."""
    user = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    main.runtime_data = IdentityRecognition(user.id)
    entity = _entity(hass, main)
    source = BlockingSource()
    source.continue_reading.set()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        await entity.async_process_audio_stream(_metadata(), _stream())
    assert take_correlated_recognition() is not None

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=None,
    ):
        result = await entity.async_process_audio_stream(_metadata(), _stream())

    assert result.result is stt.SpeechResultState.ERROR
    assert take_correlated_recognition() is None


@pytest.mark.asyncio
async def test_source_reappearing_after_missing_turn_recovers_immediately(
    hass: HomeAssistant,
) -> None:
    """Re-enabling a source requires no stale proxy reconstruction to recover."""
    user = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    main.runtime_data = IdentityRecognition(user.id)
    entity = _entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=None,
    ):
        failed = await entity.async_process_audio_stream(_metadata(), _stream())
    assert failed.result is stt.SpeechResultState.ERROR

    source = BlockingSource()
    source.continue_reading.set()
    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        recovered = await entity.async_process_audio_stream(_metadata(), _stream())

    correlated = take_correlated_recognition()
    assert recovered.result is stt.SpeechResultState.SUCCESS
    assert correlated is not None and correlated.user_id == user.id


@pytest.mark.asyncio
async def test_runtime_replacement_affects_next_turn_not_previous_completed_turn(
    hass: HomeAssistant,
) -> None:
    """A newly loaded runtime is adopted cleanly by the following utterance."""
    first_user = await hass.auth.async_create_user("First")
    second_user = await hass.auth.async_create_user("Second")
    main = _main_entry(hass)
    first_runtime = IdentityRecognition(first_user.id)
    second_runtime = IdentityRecognition(second_user.id)
    main.runtime_data = first_runtime
    entity = _entity(hass, main)
    source = BlockingSource()
    source.continue_reading.set()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        await entity.async_process_audio_stream(_metadata(), _stream(1))
        first = take_correlated_recognition()
        main.runtime_data = second_runtime
        await entity.async_process_audio_stream(_metadata(), _stream(2))
        second = take_correlated_recognition()

    assert first is not None and first.user_id == first_user.id
    assert second is not None and second.user_id == second_user.id
    assert first.utterance_sequence + 1 == second.utterance_sequence
