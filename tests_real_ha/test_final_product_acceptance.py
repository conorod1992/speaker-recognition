"""Final Real Home Assistant acceptance tests for the assembled product path."""

from __future__ import annotations

from collections.abc import AsyncIterable
from unittest.mock import patch

import pytest
from homeassistant.components import conversation, stt
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers.intent import IntentResponse
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_STT_ENTITY,
    CONF_USE_BASIC_DSP,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
)
from custom_components.speaker_recognition.conversation import (
    SpeakerRecognitionConversationEntity,
)
from custom_components.speaker_recognition.correlation import clear_correlated_recognition
from custom_components.speaker_recognition.recognition import RecognitionResult
from custom_components.speaker_recognition.stt import SpeakerRecognitionSTTEntity
from custom_components.speaker_recognition.telemetry import get_decision_history

STT_SOURCE = "stt.final_product_source"
CONVERSATION_SOURCE = "conversation.final_product_source"
THRESHOLD = 0.75


def _metadata() -> stt.SpeechMetadata:
    return stt.SpeechMetadata(
        language="en-US",
        format=stt.AudioFormats.WAV,
        codec=stt.AudioCodecs.PCM,
        sample_rate=stt.AudioSampleRates.SAMPLERATE_16000,
        channel=stt.AudioChannels.CHANNEL_MONO,
        bit_rate=stt.AudioBitRates.BITRATE_16,
    )


async def _audio_stream(marker: int = 1) -> AsyncIterable[bytes]:
    yield bytes([marker, 0]) * 8000
    yield bytes([marker, 0]) * 8000


class SourceSTT:
    """External STT boundary used by the complete HA turn."""

    supported_languages = ["en-US"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    def __init__(self, text: str = "turn on the kitchen light") -> None:
        self.text = text
        self.received = b""

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        assert metadata == _metadata()
        self.received = b"".join([chunk async for chunk in stream])
        return stt.SpeechResult(self.text, stt.SpeechResultState.SUCCESS)


class IdentityRecognition:
    """Deterministic recognition backend boundary for one runtime generation."""

    def __init__(self, user_id: str | None) -> None:
        self.user_id = user_id
        self.calls = 0

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        del audio_data, sample_rate
        self.calls += 1
        if self.user_id is None:
            return None
        return RecognitionResult(
            engine_id="resemblyzer",
            user_id=self.user_id,
            candidate_user_id=self.user_id,
            confidence=0.95,
            similarity=0.91,
            margin=0.32,
            accepted=True,
            all_scores={self.user_id: 0.91},
        )

    def pop_authoritative_diagnostics(self, audio_data: bytes):
        del audio_data
        return None


class RecordingConversationAgent:
    """External Conversation provider boundary at the end of the full turn."""

    supported_languages = ["en"]

    def __init__(self) -> None:
        self.received: list[conversation.ConversationInput] = []

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        self.received.append(user_input)
        response = IntentResponse(language=user_input.language)
        response.async_set_speech(f"handled: {user_input.text}")
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
            continue_conversation=True,
        )


def _entries(
    hass: HomeAssistant, recognition: IdentityRecognition
) -> tuple[MockConfigEntry, MockConfigEntry, MockConfigEntry]:
    main = MockConfigEntry(
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
    stt_entry = MockConfigEntry(
        domain=DOMAIN,
        title="STT proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_STT,
            CONF_STT_ENTITY: STT_SOURCE,
            CONF_USE_BASIC_DSP: False,
        },
        unique_id=f"{ENTRY_TYPE_STT}_{STT_SOURCE}",
        version=2,
        minor_version=0,
    )
    conversation_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Conversation proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: CONVERSATION_SOURCE,
            CONF_MIN_CONFIDENCE: THRESHOLD,
        },
        unique_id=f"{ENTRY_TYPE_CONVERSATION}_{CONVERSATION_SOURCE}",
        version=2,
        minor_version=0,
    )
    for entry in (main, stt_entry, conversation_entry):
        entry.add_to_hass(hass)
    main.runtime_data = recognition
    return main, stt_entry, conversation_entry


def _product_entities(
    hass: HomeAssistant,
    main: MockConfigEntry,
    stt_entry: MockConfigEntry,
    conversation_entry: MockConfigEntry,
) -> tuple[SpeakerRecognitionSTTEntity, SpeakerRecognitionConversationEntity]:
    stt_entity = SpeakerRecognitionSTTEntity(
        hass,
        stt_entry.title,
        STT_SOURCE,
        stt_entry.entry_id,
        main,
    )
    stt_entity.hass = hass
    conversation_entity = SpeakerRecognitionConversationEntity(
        hass,
        conversation_entry.title,
        CONVERSATION_SOURCE,
        conversation_entry.entry_id,
        conversation_entry,
        main,
    )
    conversation_entity.hass = hass
    return stt_entity, conversation_entity


async def _run_full_turn(
    hass: HomeAssistant,
    stt_entity: SpeakerRecognitionSTTEntity,
    conversation_entity: SpeakerRecognitionConversationEntity,
    *,
    marker: int = 1,
    context_user_id: str | None = None,
    conversation_id: str = "full-product-turn",
) -> tuple[stt.SpeechResult, conversation.ConversationResult, RecordingConversationAgent]:
    source_stt = SourceSTT()
    source_conversation = RecordingConversationAgent()
    with (
        patch(
            "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
            return_value=source_stt,
        ),
        patch(
            "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
            return_value=source_conversation,
        ),
    ):
        speech = await stt_entity.async_process_audio_stream(
            _metadata(), _audio_stream(marker)
        )
        result = await conversation_entity.async_process(
            conversation.ConversationInput(
                text=speech.text or "",
                context=Context(user_id=context_user_id),
                conversation_id=conversation_id,
                device_id="device-1",
                satellite_id="assist_satellite.kitchen",
                language="en",
                agent_id=CONVERSATION_SOURCE,
            )
        )
    await hass.async_block_till_done()
    return speech, result, source_conversation


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_recognized_voice_flows_from_stt_through_conversation_and_history(
    hass: HomeAssistant,
) -> None:
    """A real HA turn carries one recognized identity across the assembled product path."""
    user = await hass.auth.async_create_user("Alice")
    assert await async_setup_component(hass, DOMAIN, {})
    recognition = IdentityRecognition(user.id)
    main, stt_entry, conversation_entry = _entries(hass, recognition)
    stt_entity, conversation_entity = _product_entities(
        hass, main, stt_entry, conversation_entry
    )

    speech, result, source = await _run_full_turn(
        hass, stt_entity, conversation_entity
    )

    assert speech.result is stt.SpeechResultState.SUCCESS
    assert speech.text == "turn on the kitchen light"
    assert result.conversation_id == "full-product-turn"
    assert result.continue_conversation is True
    assert source.received[-1].text == speech.text
    assert source.received[-1].context.user_id == user.id
    assert recognition.calls == 1

    history = get_decision_history(hass)
    assert history is not None
    records = history.recent(25)
    assert len(records) == 1
    assert records[0]["user_id"] == user.id
    assert records[0]["accepted"] is True


@pytest.mark.asyncio
async def test_unknown_voice_remains_anonymous_through_complete_product_path(
    hass: HomeAssistant,
) -> None:
    """An unknown speaker reaches Conversation successfully without gaining HA identity."""
    assert await async_setup_component(hass, DOMAIN, {})
    recognition = IdentityRecognition(None)
    main, stt_entry, conversation_entry = _entries(hass, recognition)
    stt_entity, conversation_entity = _product_entities(
        hass, main, stt_entry, conversation_entry
    )

    speech, result, source = await _run_full_turn(
        hass, stt_entity, conversation_entity
    )

    assert speech.result is stt.SpeechResultState.SUCCESS
    assert result.conversation_id == "full-product-turn"
    assert source.received[-1].context.user_id is None
    history = get_decision_history(hass)
    assert history is not None
    records = history.recent(25)
    assert len(records) == 1
    assert records[0]["accepted"] is False


@pytest.mark.asyncio
async def test_authenticated_context_wins_after_real_stt_recognizes_different_user(
    hass: HomeAssistant,
) -> None:
    """End-to-end recognition cannot replace identity already authenticated by HA."""
    recognized = await hass.auth.async_create_user("Recognized")
    authenticated = await hass.auth.async_create_user("Authenticated")
    recognition = IdentityRecognition(recognized.id)
    main, stt_entry, conversation_entry = _entries(hass, recognition)
    stt_entity, conversation_entity = _product_entities(
        hass, main, stt_entry, conversation_entry
    )

    _, _, source = await _run_full_turn(
        hass,
        stt_entity,
        conversation_entity,
        context_user_id=authenticated.id,
    )

    assert source.received[-1].context.user_id == authenticated.id


@pytest.mark.asyncio
async def test_user_deleted_between_stt_and_conversation_fails_closed(
    hass: HomeAssistant,
) -> None:
    """A user disappearing after recognition cannot be injected by the same turn."""
    await hass.auth.async_create_user("Owner")
    user = await hass.auth.async_create_user("Alice")
    assert not user.is_owner
    recognition = IdentityRecognition(user.id)
    main, stt_entry, conversation_entry = _entries(hass, recognition)
    stt_entity, conversation_entity = _product_entities(
        hass, main, stt_entry, conversation_entry
    )
    source_stt = SourceSTT()
    source_conversation = RecordingConversationAgent()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source_stt,
    ):
        speech = await stt_entity.async_process_audio_stream(_metadata(), _audio_stream())

    await hass.auth.async_remove_user(user)

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source_conversation,
    ):
        await conversation_entity.async_process(
            conversation.ConversationInput(
                text=speech.text or "",
                context=Context(),
                conversation_id="deleted-mid-turn",
                device_id="device-1",
                satellite_id="assist_satellite.kitchen",
                language="en",
                agent_id=CONVERSATION_SOURCE,
            )
        )

    assert source_conversation.received[-1].context.user_id is None


@pytest.mark.asyncio
async def test_runtime_generation_changes_cleanly_between_complete_turns(
    hass: HomeAssistant,
) -> None:
    """A main-runtime replacement changes the next full turn without identity leakage."""
    first_user = await hass.auth.async_create_user("First")
    second_user = await hass.auth.async_create_user("Second")
    first_runtime = IdentityRecognition(first_user.id)
    second_runtime = IdentityRecognition(second_user.id)
    main, stt_entry, conversation_entry = _entries(hass, first_runtime)
    stt_entity, conversation_entity = _product_entities(
        hass, main, stt_entry, conversation_entry
    )

    _, _, first_source = await _run_full_turn(
        hass,
        stt_entity,
        conversation_entity,
        marker=1,
        conversation_id="turn-1",
    )
    main.runtime_data = second_runtime
    _, _, second_source = await _run_full_turn(
        hass,
        stt_entity,
        conversation_entity,
        marker=2,
        conversation_id="turn-2",
    )

    assert first_source.received[-1].context.user_id == first_user.id
    assert second_source.received[-1].context.user_id == second_user.id
    assert first_runtime.calls == 1
    assert second_runtime.calls == 1
