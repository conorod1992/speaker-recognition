"""Real Home Assistant acceptance tests for the STT proxy."""

from __future__ import annotations

from collections.abc import AsyncIterable
from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.components import stt
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_STT_ENTITY,
    CONF_USE_BASIC_DSP,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
)
from custom_components.speaker_recognition.correlation import (
    clear_correlated_recognition,
    take_correlated_recognition,
)
from custom_components.speaker_recognition.recognition import RecognitionResult
from custom_components.speaker_recognition.stt import SpeakerRecognitionSTTEntity


SOURCE_ENTITY_ID = "stt.test_source"


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


async def _open_stt_flow(hass: HomeAssistant) -> dict:
    _main_entry(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "menu"
    return await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_stt"}
    )


def _metadata() -> stt.SpeechMetadata:
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
        bit_rate=stt.AudioBitRates.BITRATE_16,
    )


async def _audio_stream() -> AsyncIterable[bytes]:
    yield b"\x00\x00" * 8000
    yield b"\x01\x00" * 8000


class FakeSourceSTT:
    """Minimal external STT provider boundary for proxy acceptance tests."""

    supported_languages = ["en-US"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.received = b""

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        assert metadata == _metadata()
        self.received = b"".join([chunk async for chunk in stream])
        if self.fail:
            raise RuntimeError("source STT failed")
        return stt.SpeechResult("hello from source", stt.SpeechResultState.SUCCESS)


class FakeRecognition:
    """Small recognition boundary used by the real proxy entity."""

    def __init__(self, result: RecognitionResult | None) -> None:
        self.result = result
        self.received: tuple[bytes, int] | None = None

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        self.received = (audio_data, sample_rate)
        return self.result

    def pop_authoritative_diagnostics(self, audio_data: bytes):
        del audio_data
        return None


def _proxy_entity(
    hass: HomeAssistant, main: MockConfigEntry
) -> SpeakerRecognitionSTTEntity:
    """Create a proxy entity attached to the real HA instance for direct execution."""
    entity = SpeakerRecognitionSTTEntity(
        hass, "Speaker Recognition STT", SOURCE_ENTITY_ID, "proxy-id", main
    )
    # EntityPlatform normally supplies this reference before an entity is used. These
    # tests exercise the real proxy class directly, so attach the same HA instance here
    # without mocking any Speaker Recognition behavior.
    entity.hass = hass
    return entity


@pytest.mark.asyncio
async def test_real_config_flow_creates_stt_proxy_entry(hass: HomeAssistant) -> None:
    """The genuine HA config flow creates a current-version STT proxy entry."""
    result = await _open_stt_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "add_stt"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_STT_ENTITY: SOURCE_ENTITY_ID, CONF_USE_BASIC_DSP: False},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    entry = result["result"]
    assert entry.version == 2
    assert entry.data[CONF_ENTRY_TYPE] == ENTRY_TYPE_STT
    assert entry.data[CONF_STT_ENTITY] == SOURCE_ENTITY_ID
    assert entry.unique_id == f"{ENTRY_TYPE_STT}_{SOURCE_ENTITY_ID}"


@pytest.mark.asyncio
async def test_stt_flow_rejects_duplicate_wrapped_source(hass: HomeAssistant) -> None:
    """A source STT entity cannot be wrapped twice."""
    _main_entry(hass)
    existing = MockConfigEntry(
        domain=DOMAIN,
        title="Existing STT proxy",
        data={CONF_ENTRY_TYPE: ENTRY_TYPE_STT, CONF_STT_ENTITY: SOURCE_ENTITY_ID},
        unique_id=f"{ENTRY_TYPE_STT}_{SOURCE_ENTITY_ID}",
        version=2,
        minor_version=0,
    )
    existing.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_stt"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_STT_ENTITY: SOURCE_ENTITY_ID, CONF_USE_BASIC_DSP: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "proxy_source_already_wrapped"}


@pytest.mark.asyncio
async def test_stt_flow_rejects_speaker_recognition_proxy_as_source(
    hass: HomeAssistant,
) -> None:
    """Recursive Speaker Recognition proxy chains are rejected."""
    _main_entry(hass)
    wrapped_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Existing STT proxy",
        data={CONF_ENTRY_TYPE: ENTRY_TYPE_STT, CONF_STT_ENTITY: "stt.original"},
        unique_id=f"{ENTRY_TYPE_STT}_stt.original",
        version=2,
        minor_version=0,
    )
    wrapped_entry.add_to_hass(hass)
    registry = er.async_get(hass)
    recursive_source = registry.async_get_or_create(
        "stt",
        DOMAIN,
        "existing-proxy",
        suggested_object_id="speaker_recognition_proxy",
        config_entry=wrapped_entry,
    ).entity_id

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_stt"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_STT_ENTITY: recursive_source, CONF_USE_BASIC_DSP: False},
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "speaker_recognition_proxy_source"}


@pytest.mark.asyncio
async def test_stt_proxy_forwards_audio_and_preserves_transcription(
    hass: HomeAssistant,
) -> None:
    """The real proxy entity forwards the complete stream to its source STT entity."""
    main = _main_entry(hass)
    recognition = FakeRecognition(None)
    main.runtime_data = recognition
    source = FakeSourceSTT()
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        result = await entity.async_process_audio_stream(_metadata(), _audio_stream())

    assert result.text == "hello from source"
    assert result.result is stt.SpeechResultState.SUCCESS
    assert source.received == (b"\x00\x00" * 8000) + (b"\x01\x00" * 8000)
    assert recognition.received is not None
    assert recognition.received[1] == 16000


@pytest.mark.asyncio
async def test_stt_proxy_correlates_recognized_user_with_exact_turn(
    hass: HomeAssistant,
) -> None:
    """A recognized user is attached to the same HA STT task's correlation context."""
    clear_correlated_recognition()
    user = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    main.runtime_data = FakeRecognition(
        RecognitionResult(
            engine_id="resemblyzer",
            user_id=user.id,
            candidate_user_id=user.id,
            confidence=0.94,
            similarity=0.88,
            margin=0.31,
            accepted=True,
            all_scores={user.id: 0.88},
        )
    )
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=FakeSourceSTT(),
    ):
        result = await entity.async_process_audio_stream(_metadata(), _audio_stream())

    assert result.text == "hello from source"
    correlated = take_correlated_recognition()
    assert correlated is not None
    assert correlated.user_id == user.id
    assert correlated.accepted is True
    assert correlated.candidate_user_id == user.id
    assert correlated.utterance_sequence == 1


@pytest.mark.asyncio
async def test_unknown_speaker_does_not_create_identity_context(
    hass: HomeAssistant,
) -> None:
    """No recognition result creates a fail-closed, anonymous turn correlation."""
    clear_correlated_recognition()
    main = _main_entry(hass)
    main.runtime_data = FakeRecognition(None)
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=FakeSourceSTT(),
    ):
        await entity.async_process_audio_stream(_metadata(), _audio_stream())

    correlated = take_correlated_recognition()
    assert correlated is not None
    assert correlated.user_id is None
    assert correlated.accepted is False
    assert correlated.candidate_user_id == ""


@pytest.mark.asyncio
async def test_missing_source_stt_returns_clean_error(hass: HomeAssistant) -> None:
    """A source entity disappearing after configuration fails the turn cleanly."""
    main = _main_entry(hass)
    main.runtime_data = FakeRecognition(None)
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=None,
    ):
        result = await entity.async_process_audio_stream(_metadata(), _audio_stream())

    assert result.text is None
    assert result.result is stt.SpeechResultState.ERROR


@pytest.mark.asyncio
async def test_underlying_stt_failure_propagates_without_stale_identity(
    hass: HomeAssistant,
) -> None:
    """Source STT failure propagates and cannot leave a previous speaker attached."""
    clear_correlated_recognition()
    main = _main_entry(hass)
    main.runtime_data = FakeRecognition(None)
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=FakeSourceSTT(fail=True),
    ):
        with pytest.raises(RuntimeError, match="source STT failed"):
            await entity.async_process_audio_stream(_metadata(), _audio_stream())

    assert take_correlated_recognition() is None
