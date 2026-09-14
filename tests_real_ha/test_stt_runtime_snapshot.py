"""Regression coverage for STT recognition runtime replacement during a turn."""

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
    CorrelatedRecognition,
    clear_correlated_recognition,
    take_correlated_recognition,
)
from custom_components.speaker_recognition.recognition import RecognitionResult
from custom_components.speaker_recognition.stt import SpeakerRecognitionSTTEntity

SOURCE = "stt.runtime_snapshot_source"


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
        "Runtime snapshot proxy",
        SOURCE,
        "runtime-snapshot-proxy",
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


async def _stream() -> AsyncIterable[bytes]:
    yield b"\x01\x00" * 4000
    await asyncio.sleep(0)
    yield b"\x01\x00" * 4000


class BlockingSourceSTT:
    """Pause source STT after the first chunk to expose the reload window."""

    supported_languages = ["en-US"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    def __init__(self, *, block: bool = True) -> None:
        self.first_chunk_seen = asyncio.Event()
        self.continue_reading = asyncio.Event()
        if not block:
            self.continue_reading.set()

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


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_inflight_turn_keeps_starting_runtime_and_next_turn_uses_replacement(
    hass: HomeAssistant,
) -> None:
    """Reloading the main runtime mid-stream must not split one utterance generations."""
    old_user = await hass.auth.async_create_user("Old runtime")
    new_user = await hass.auth.async_create_user("New runtime")
    main = _main_entry(hass)
    old_runtime = IdentityRecognition(old_user.id)
    new_runtime = IdentityRecognition(new_user.id)
    main.runtime_data = old_runtime
    entity = _entity(hass, main)
    blocked_source = BlockingSourceSTT()

    async def run_first_turn() -> tuple[stt.SpeechResult, CorrelatedRecognition | None]:
        result = await entity.async_process_audio_stream(_metadata(), _stream())
        return result, take_correlated_recognition()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=blocked_source,
    ):
        task = asyncio.create_task(run_first_turn())
        await blocked_source.first_chunk_seen.wait()
        main.runtime_data = new_runtime
        blocked_source.continue_reading.set()
        first_result, first = await task

    assert first_result.result is stt.SpeechResultState.SUCCESS
    assert first is not None and first.user_id == old_user.id
    assert old_runtime.calls == 1
    assert new_runtime.calls == 0

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=BlockingSourceSTT(block=False),
    ):
        second_result = await entity.async_process_audio_stream(_metadata(), _stream())

    second = take_correlated_recognition()
    assert second_result.result is stt.SpeechResultState.SUCCESS
    assert second is not None and second.user_id == new_user.id
    assert old_runtime.calls == 1
    assert new_runtime.calls == 1
    assert second.utterance_sequence == first.utterance_sequence + 1
