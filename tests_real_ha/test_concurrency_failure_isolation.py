"""Real Home Assistant acceptance tests for concurrent turn isolation and recovery."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from unittest.mock import patch

import pytest
from homeassistant.components import conversation, stt
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers.intent import IntentResponse
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
)
from custom_components.speaker_recognition.conversation import (
    SpeakerRecognitionConversationEntity,
)
from custom_components.speaker_recognition.correlation import (
    CorrelatedRecognition,
    clear_correlated_recognition,
    set_correlated_recognition,
    take_correlated_recognition,
)
from custom_components.speaker_recognition.recognition import RecognitionResult
from custom_components.speaker_recognition.stt import SpeakerRecognitionSTTEntity


STT_SOURCE = "stt.concurrent_source"
CONVERSATION_SOURCE = "conversation.concurrent_source"


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


def _stt_entity(hass: HomeAssistant, main: MockConfigEntry) -> SpeakerRecognitionSTTEntity:
    entity = SpeakerRecognitionSTTEntity(
        hass,
        "Concurrent STT proxy",
        STT_SOURCE,
        "concurrent-stt-proxy",
        main,
    )
    entity.hass = hass
    return entity


def _conversation_entry(
    hass: HomeAssistant, *, threshold: float = 0.75
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Concurrent Conversation proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: CONVERSATION_SOURCE,
            CONF_MIN_CONFIDENCE: threshold,
        },
        unique_id=f"{ENTRY_TYPE_CONVERSATION}_{CONVERSATION_SOURCE}",
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


def _conversation_entity(
    hass: HomeAssistant,
    main: MockConfigEntry,
    entry: MockConfigEntry | None = None,
) -> SpeakerRecognitionConversationEntity:
    entry = entry or _conversation_entry(hass)
    entity = SpeakerRecognitionConversationEntity(
        hass,
        entry.title,
        CONVERSATION_SOURCE,
        entry.entry_id,
        entry,
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


async def _stream(marker: int) -> AsyncIterable[bytes]:
    yield bytes([marker, 0]) * 4000
    await asyncio.sleep(0)
    yield bytes([marker, 0]) * 4000


class ConcurrentSourceSTT:
    """External STT boundary that deliberately yields to concurrent turns."""

    supported_languages = ["en-US"]
    supported_formats = [stt.AudioFormats.WAV]
    supported_codecs = [stt.AudioCodecs.PCM]
    supported_bit_rates = [stt.AudioBitRates.BITRATE_16]
    supported_sample_rates = [stt.AudioSampleRates.SAMPLERATE_16000]
    supported_channels = [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        del metadata
        async for _chunk in stream:
            await asyncio.sleep(0)
        return stt.SpeechResult("ok", stt.SpeechResultState.SUCCESS)


class EarlyReturnSourceSTT(ConcurrentSourceSTT):
    """Provider that returns before fully consuming the supplied stream."""

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        del metadata
        async for _chunk in stream:
            break
        return stt.SpeechResult("partial", stt.SpeechResultState.SUCCESS)


class MarkerRecognition:
    """Return deterministic identities from the first PCM sample byte."""

    def __init__(self, users: dict[int, str], *, fail_marker: int | None = None) -> None:
        self.users = users
        self.fail_marker = fail_marker

    async def async_recognize(
        self, audio_data: bytes, sample_rate: int = 16000
    ) -> RecognitionResult | None:
        del sample_rate
        await asyncio.sleep(0)
        marker = audio_data[0]
        if marker == self.fail_marker:
            raise RuntimeError("recognition failed")
        user_id = self.users.get(marker)
        if user_id is None:
            return None
        return RecognitionResult(
            engine_id="resemblyzer",
            user_id=user_id,
            candidate_user_id=user_id,
            confidence=0.95,
            similarity=0.90,
            margin=0.30,
            accepted=True,
            all_scores={user_id: 0.90},
        )

    def pop_authoritative_diagnostics(self, audio_data: bytes):
        del audio_data
        return None


class RecordingConversationAgent:
    """Conversation boundary that records inputs while allowing overlap."""

    supported_languages = ["en"]

    def __init__(self, *, fail_text: str | None = None) -> None:
        self.fail_text = fail_text
        self.received: list[conversation.ConversationInput] = []

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        self.received.append(user_input)
        await asyncio.sleep(0)
        if user_input.text == self.fail_text:
            raise RuntimeError("conversation failed")
        response = IntentResponse(language=user_input.language)
        response.async_set_speech(user_input.text)
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )


def _conversation_input(text: str) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text=text,
        context=Context(),
        conversation_id=f"conversation-{text}",
        device_id="device-1",
        satellite_id="assist_satellite.kitchen",
        language="en",
        agent_id=CONVERSATION_SOURCE,
    )


def _correlated(user_id: str, sequence: int) -> CorrelatedRecognition:
    return CorrelatedRecognition(
        user_id=user_id,
        candidate_user_id=user_id,
        confidence=0.95,
        similarity=0.90,
        margin=0.30,
        accepted=True,
        all_scores={user_id: 0.90},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=sequence,
    )


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_two_simultaneous_stt_turns_keep_speaker_context_task_local(
    hass: HomeAssistant,
) -> None:
    """Overlapping STT requests cannot exchange recognized HA users."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    main.runtime_data = MarkerRecognition({1: alice.id, 2: bob.id})
    entity = _stt_entity(hass, main)
    source = ConcurrentSourceSTT()

    async def run(marker: int) -> CorrelatedRecognition | None:
        await entity.async_process_audio_stream(_metadata(), _stream(marker))
        return take_correlated_recognition()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=source,
    ):
        first, second = await asyncio.gather(run(1), run(2))

    assert first is not None and first.user_id == alice.id
    assert second is not None and second.user_id == bob.id
    assert first.utterance_sequence != second.utterance_sequence


@pytest.mark.asyncio
async def test_concurrent_stt_turns_allocate_unique_monotonic_sequences(
    hass: HomeAssistant,
) -> None:
    """Concurrent turns receive distinct correlation IDs with no lost increment."""
    main = _main_entry(hass)
    main.runtime_data = MarkerRecognition({})
    entity = _stt_entity(hass, main)

    async def run(marker: int) -> int:
        await entity.async_process_audio_stream(_metadata(), _stream(marker))
        correlated = take_correlated_recognition()
        assert correlated is not None
        return correlated.utterance_sequence

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=ConcurrentSourceSTT(),
    ):
        sequences = await asyncio.gather(*(run(marker) for marker in range(1, 7)))

    assert sorted(sequences) == list(range(1, 7))
    assert hass.data[DOMAIN]["utterance_sequence"] == 6


@pytest.mark.asyncio
async def test_recognition_failure_in_one_stt_turn_does_not_poison_sibling(
    hass: HomeAssistant,
) -> None:
    """Supplemental recognition failure remains isolated to its own Assist task."""
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    main.runtime_data = MarkerRecognition({2: bob.id}, fail_marker=1)
    entity = _stt_entity(hass, main)

    async def run(marker: int) -> CorrelatedRecognition | None:
        result = await entity.async_process_audio_stream(_metadata(), _stream(marker))
        assert result.result is stt.SpeechResultState.SUCCESS
        return take_correlated_recognition()

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=ConcurrentSourceSTT(),
    ):
        failed, healthy = await asyncio.gather(run(1), run(2))

    assert failed is not None
    assert failed.user_id is None
    assert failed.accepted is False
    assert healthy is not None
    assert healthy.user_id == bob.id
    assert healthy.accepted is True


@pytest.mark.asyncio
async def test_source_returning_before_eof_skips_recognition_without_stale_identity(
    hass: HomeAssistant,
) -> None:
    """An STT provider that abandons the stream cannot correlate partial audio."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    main.runtime_data = MarkerRecognition({1: alice.id})
    entity = _stt_entity(hass, main)

    set_correlated_recognition(_correlated(alice.id, 99))
    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=EarlyReturnSourceSTT(),
    ):
        result = await entity.async_process_audio_stream(_metadata(), _stream(1))

    correlated = take_correlated_recognition()
    assert result.text == "partial"
    assert correlated is not None
    assert correlated.user_id is None
    assert correlated.accepted is False


@pytest.mark.asyncio
async def test_stt_turn_recovers_immediately_after_recognition_exception(
    hass: HomeAssistant,
) -> None:
    """A failed recognition call does not leave the next turn unhealthy."""
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    main.runtime_data = MarkerRecognition({2: bob.id}, fail_marker=1)
    entity = _stt_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.stt.async_get_speech_to_text_entity",
        return_value=ConcurrentSourceSTT(),
    ):
        await entity.async_process_audio_stream(_metadata(), _stream(1))
        first = take_correlated_recognition()
        await entity.async_process_audio_stream(_metadata(), _stream(2))
        second = take_correlated_recognition()

    assert first is not None and first.user_id is None
    assert second is not None and second.user_id == bob.id
    assert second.utterance_sequence == first.utterance_sequence + 1


@pytest.mark.asyncio
async def test_two_simultaneous_conversation_turns_keep_identity_task_local(
    hass: HomeAssistant,
) -> None:
    """Concurrent Conversation proxy calls cannot exchange recognized users."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    entity = _conversation_entity(hass, main)
    source = RecordingConversationAgent()

    async def run(text: str, user_id: str, sequence: int) -> str | None:
        set_correlated_recognition(_correlated(user_id, sequence))
        with patch(
            "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
            return_value=source,
        ):
            await entity.async_process(_conversation_input(text))
        matching = next(item for item in source.received if item.text == text)
        return matching.context.user_id

    first, second = await asyncio.gather(
        run("alice", alice.id, 1),
        run("bob", bob.id, 2),
    )

    assert first == alice.id
    assert second == bob.id


@pytest.mark.asyncio
async def test_concurrent_conversation_failure_does_not_contaminate_other_turn(
    hass: HomeAssistant,
) -> None:
    """One failing source agent call cannot consume or replace a sibling identity."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    entity = _conversation_entity(hass, main)
    source = RecordingConversationAgent(fail_text="fail")

    async def failing_turn() -> None:
        set_correlated_recognition(_correlated(alice.id, 1))
        with patch(
            "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
            return_value=source,
        ):
            with pytest.raises(RuntimeError, match="conversation failed"):
                await entity.async_process(_conversation_input("fail"))

    async def healthy_turn() -> str | None:
        set_correlated_recognition(_correlated(bob.id, 2))
        with patch(
            "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
            return_value=source,
        ):
            await entity.async_process(_conversation_input("healthy"))
        matching = next(item for item in source.received if item.text == "healthy")
        return matching.context.user_id

    _, healthy_user = await asyncio.gather(failing_turn(), healthy_turn())
    assert healthy_user == bob.id


@pytest.mark.asyncio
async def test_conversation_threshold_option_change_applies_to_next_turn_only(
    hass: HomeAssistant,
) -> None:
    """Live option changes are observed by subsequent turns without stale policy."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    entry = _conversation_entry(hass, threshold=0.75)
    entity = _conversation_entity(hass, main, entry)
    source = RecordingConversationAgent()

    recognition = CorrelatedRecognition(
        user_id=alice.id,
        candidate_user_id=alice.id,
        confidence=0.80,
        similarity=0.80,
        margin=0.20,
        accepted=True,
        all_scores={alice.id: 0.80},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=1,
    )

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        set_correlated_recognition(recognition)
        await entity.async_process(_conversation_input("before"))

        hass.config_entries.async_update_entry(entry, options={CONF_MIN_CONFIDENCE: 0.90})

        set_correlated_recognition(recognition)
        await entity.async_process(_conversation_input("after"))

    before = next(item for item in source.received if item.text == "before")
    after = next(item for item in source.received if item.text == "after")
    assert before.context.user_id == alice.id
    assert after.context.user_id is None
