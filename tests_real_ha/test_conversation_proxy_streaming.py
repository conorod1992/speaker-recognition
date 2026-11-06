"""Real Assist pipeline acceptance tests for Conversation proxy streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
import inspect
from typing import Any
from unittest.mock import patch

import pytest
from homeassistant.components import conversation, tts
from homeassistant.components.assist_pipeline.models import PipelineEvent, PipelineStage
from homeassistant.components.assist_pipeline.pipeline import (
    Pipeline,
    PipelineInput,
    PipelineRun,
)
from homeassistant.components.conversation.const import DATA_COMPONENT
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import Unauthorized
from homeassistant.helpers import chat_session
from homeassistant.helpers.intent import IntentResponse
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
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
)

SOURCE_ENTITY_ID = "conversation.source"
PROXY_ENTITY_ID = "conversation.proxy"
TTS_ENTITY_ID = "tts.streaming"
# The pipeline starts streaming TTS after more than 60 characters.
FIRST_PART = "This first part of the answer is longer than the streaming threshold. "
SECOND_PART = "This second part only exists after TTS streaming has started."

# Home Assistant 2026.11 made chat sessions owned by a user and made the context
# argument mandatory. Earlier versions do not accept that argument.
SESSIONS_HAVE_OWNER = (
    "context" in inspect.signature(chat_session.async_get_chat_session).parameters
)
SESSION_ARGS: dict[str, Any] = {"context": None} if SESSIONS_HAVE_OWNER else {}


class StreamingSource(conversation.ConversationEntity):
    """Source agent that blocks mid-stream until the pipeline starts TTS.

    A proxy that buffered the answer until the agent finished would never let
    the pipeline start TTS, so this agent would time out instead of finishing.
    """

    _attr_name = "Source"
    _attr_unique_id = "source"
    _attr_supports_streaming = True

    def __init__(self) -> None:
        self.entity_id = SOURCE_ENTITY_ID
        self.tts_started = asyncio.Event()
        self.user_id_at_first_delta: str | None = None

    @property
    def supported_languages(self) -> str:
        return "*"

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        async def deltas() -> AsyncGenerator[dict[str, str]]:
            yield {"role": "assistant"}
            self.user_id_at_first_delta = user_input.context.user_id
            yield {"content": FIRST_PART}
            await asyncio.wait_for(self.tts_started.wait(), 5)
            yield {"content": SECOND_PART}

        async for _ in chat_log.async_add_delta_content_stream(
            user_input.agent_id, deltas()
        ):
            pass
        return conversation.async_get_result_from_chat_log(user_input, chat_log)


class NonStreamingSource(conversation.ConversationEntity):
    """Source agent that only answers with its final result."""

    _attr_name = "Source"
    _attr_unique_id = "source"

    def __init__(self) -> None:
        self.entity_id = SOURCE_ENTITY_ID

    @property
    def supported_languages(self) -> str:
        return "*"

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        response = IntentResponse(language=user_input.language)
        response.async_set_speech(FIRST_PART + SECOND_PART)
        return conversation.ConversationResult(
            response=response, conversation_id=user_input.conversation_id
        )


class StreamingTts(tts.TextToSpeechEntity):
    """TTS entity that accepts streaming input, as the pipeline requires."""

    _attr_name = "Streaming"
    _attr_unique_id = "streaming"
    _attr_supported_languages = ["en"]
    _attr_default_language = "en"

    def __init__(self) -> None:
        self.entity_id = TTS_ENTITY_ID

    def async_supports_streaming_input(self) -> bool:
        return True

    async def async_stream_tts_audio(
        self, request: tts.TTSAudioRequest
    ) -> tts.TTSAudioResponse:
        async def audio() -> AsyncGenerator[bytes]:
            async for _ in request.message_gen:
                pass
            yield b"audio"

        return tts.TTSAudioResponse("mp3", audio())


@dataclass
class Turn:
    """What the pipeline reported about one conversation turn."""

    events: list[PipelineEvent]
    conversation_id: str

    def _data(self, event_type: str) -> list[dict[str, Any]]:
        return [event.data or {} for event in self.events if event.type == event_type]

    @property
    def stream_response(self) -> bool | None:
        return self._data("run-start")[0]["tts_output"]["stream_response"]

    @property
    def tts_start_streaming(self) -> bool:
        return any(
            data.get("tts_start_streaming") for data in self._data("intent-progress")
        )

    @property
    def streamed_text(self) -> str:
        return "".join(
            data["chat_log_delta"].get("content", "")
            for data in self._data("intent-progress")
            if "chat_log_delta" in data
        )

    @property
    def errors(self) -> list[dict[str, Any]]:
        return self._data("error")

    @property
    def response(self) -> dict[str, Any]:
        return self._data("intent-end")[0]["intent_output"]["response"]


async def _setup_proxy(
    hass: HomeAssistant, source: conversation.ConversationEntity
) -> SpeakerRecognitionConversationEntity:
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "assist_pipeline", {})
    await hass.data[tts.DATA_COMPONENT].async_add_entities([StreamingTts()])

    main_entry = MockConfigEntry(
        domain=DOMAIN, data={CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN}, version=2
    )
    main_entry.add_to_hass(hass)
    proxy_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Conversation proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: SOURCE_ENTITY_ID,
            CONF_MIN_CONFIDENCE: 0.75,
        },
        version=2,
    )
    proxy_entry.add_to_hass(hass)

    proxy = SpeakerRecognitionConversationEntity(
        hass,
        proxy_entry.title,
        SOURCE_ENTITY_ID,
        proxy_entry.entry_id,
        proxy_entry,
        main_entry,
    )
    proxy.entity_id = PROXY_ENTITY_ID
    await hass.data[DATA_COMPONENT].async_add_entities([source, proxy])
    await hass.async_block_till_done()
    return proxy


async def _run_turn(
    hass: HomeAssistant,
    source: conversation.ConversationEntity,
    conversation_id: str | None = None,
) -> Turn:
    """Run one anonymous turn, like a satellite, through the proxy."""
    events: list[PipelineEvent] = []

    def on_event(event: PipelineEvent) -> None:
        events.append(event)
        if (event.data or {}).get("tts_start_streaming"):
            source.tts_started.set()

    pipeline = Pipeline(
        conversation_engine=PROXY_ENTITY_ID,
        conversation_language="*",
        language="en",
        name="Streaming test",
        stt_engine=None,
        stt_language=None,
        tts_engine=TTS_ENTITY_ID,
        tts_language="en",
        tts_voice=None,
        wake_word_entity=None,
        wake_word_id=None,
        prefer_local_intents=False,
    )
    with chat_session.async_get_chat_session(
        hass, conversation_id, **SESSION_ARGS
    ) as session:
        await PipelineInput(
            session=session,
            intent_input="tell me something long",
            run=PipelineRun(
                hass,
                context=Context(),
                pipeline=pipeline,
                start_stage=PipelineStage.INTENT,
                end_stage=PipelineStage.TTS,
                event_callback=on_event,
            ),
        ).execute(validate=True)
    await hass.async_block_till_done()
    return Turn(events, session.conversation_id)


def _recognition(user_id: str) -> CorrelatedRecognition:
    return CorrelatedRecognition(
        user_id=user_id,
        candidate_user_id=user_id,
        confidence=0.94,
        similarity=0.88,
        margin=0.31,
        accepted=True,
        all_scores={user_id: 0.88},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=1,
    )


@pytest.fixture(autouse=True)
def _clear_turn_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_proxy_passes_source_deltas_through_while_source_is_running(
    hass: HomeAssistant,
) -> None:
    """TTS starts while the source agent is still in the middle of its answer."""
    source = StreamingSource()
    await _setup_proxy(hass, source)

    turn = await _run_turn(hass, source)

    assert turn.errors == []
    assert turn.stream_response is True
    assert turn.tts_start_streaming is True
    assert turn.streamed_text == FIRST_PART + SECOND_PART
    assert turn.response["speech"]["plain"]["speech"] == FIRST_PART + SECOND_PART


@pytest.mark.asyncio
async def test_proxy_falls_back_when_source_cannot_stream(hass: HomeAssistant) -> None:
    """A non-streaming source keeps the behavior of returning one final answer."""
    source = NonStreamingSource()
    await _setup_proxy(hass, source)

    turn = await _run_turn(hass, source)

    assert turn.errors == []
    assert turn.stream_response is False
    assert turn.tts_start_streaming is False
    assert turn.streamed_text == ""
    assert turn.response["speech"]["plain"]["speech"] == FIRST_PART + SECOND_PART


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [None, object()],
    ids=["missing_source", "legacy_agent_without_streaming_property"],
)
async def test_proxy_does_not_claim_streaming_without_capable_source(
    hass: HomeAssistant, agent: object | None
) -> None:
    """Agents that are gone or predate streaming report no streaming support."""
    proxy = await _setup_proxy(hass, NonStreamingSource())

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=agent,
    ):
        assert proxy.supports_streaming is False
