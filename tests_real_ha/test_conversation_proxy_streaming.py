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
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.exceptions import Unauthorized
from homeassistant.helpers import chat_session, llm
from homeassistant.helpers.intent import IntentResponse
from homeassistant.helpers.typing import UNDEFINED
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_MIN_CONFIDENCE,
    CONF_SPEAKER_NOTE_ENABLED,
    CONF_SPEAKER_NOTE_TEMPLATE,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
)
from custom_components.speaker_recognition.conversation import (
    SpeakerRecognitionConversationEntity,
    _scoped_session_owner,
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

# Home Assistant 2026.11 added session ownership and a context argument.
# Earlier versions do not accept that argument.
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
    hass: HomeAssistant,
    source: conversation.ConversationEntity,
    *,
    options: dict[str, Any] | None = None,
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
        options=options or {},
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


@pytest.mark.asyncio
async def test_recognized_speaker_streams_with_identity_before_first_delta(
    hass: HomeAssistant,
) -> None:
    """A recognized speaker gets a streamed answer and is known to the first delta."""
    alice = await hass.auth.async_create_user("Alice")
    source = StreamingSource()
    await _setup_proxy(hass, source)
    set_correlated_recognition(_recognition(alice.id))

    turn = await _run_turn(hass, source)

    assert turn.errors == []
    assert turn.response["response_type"] == "action_done"
    assert turn.tts_start_streaming is True
    assert turn.streamed_text == FIRST_PART + SECOND_PART
    assert source.user_id_at_first_delta == alice.id


@pytest.mark.asyncio
async def test_recognized_identity_does_not_outlive_its_turn(
    hass: HomeAssistant,
) -> None:
    """Follow-up turns of a satellite conversation start anonymous again."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    source = StreamingSource()
    await _setup_proxy(hass, source)

    set_correlated_recognition(_recognition(alice.id))
    first = await _run_turn(hass, source)
    assert first.errors == []

    source.tts_started.clear()
    second = await _run_turn(hass, source, first.conversation_id)
    assert second.errors == []
    assert second.conversation_id == first.conversation_id
    assert second.response["response_type"] == "action_done"
    assert source.user_id_at_first_delta is None

    source.tts_started.clear()
    set_correlated_recognition(_recognition(bob.id))
    third = await _run_turn(hass, source, first.conversation_id)
    assert third.errors == []
    assert third.response["response_type"] == "action_done"
    assert source.user_id_at_first_delta == bob.id


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SESSIONS_HAVE_OWNER, reason="Sessions have no owner before HA 2026.11"
)
async def test_recognition_cannot_take_over_a_conversation_owned_by_a_user(
    hass: HomeAssistant,
) -> None:
    """Speaker recognition must not get past the owner check of a real user."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    source = StreamingSource()
    proxy = await _setup_proxy(hass, source)
    set_correlated_recognition(_recognition(alice.id))

    with chat_session.async_get_chat_session(
        hass, None, context=Context(user_id=bob.id)
    ) as session:
        with pytest.raises(Unauthorized):
            await proxy.async_process(
                conversation.ConversationInput(
                    text="tell me something long",
                    context=Context(),
                    conversation_id=session.conversation_id,
                    device_id=None,
                    satellite_id=None,
                    language="en",
                    agent_id=PROXY_ENTITY_ID,
                )
            )
        assert session.user_id == bob.id
    assert source.user_id_at_first_delta is None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SESSIONS_HAVE_OWNER, reason="Sessions have no owner before HA 2026.11"
)
@pytest.mark.parametrize(
    "anonymous_context", [None, Context()], ids=["unclaimed", "anonymous"]
)
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_session_owner_is_restored_after_failure(
    hass: HomeAssistant, anonymous_context, error_type
) -> None:
    """Failures and cancellation restore both kinds of anonymous session owner."""
    with chat_session.async_get_chat_session(
        hass, None, context=anonymous_context
    ) as session:
        original_owner = session.user_id
        assert original_owner is None or original_owner is UNDEFINED
        with pytest.raises(error_type):
            with _scoped_session_owner(session.conversation_id, "alice"):
                assert session.user_id == "alice"
                with chat_session.async_get_chat_session(
                    hass, session.conversation_id, context=Context(user_id="alice")
                ):
                    raise error_type()
        assert session.user_id is original_owner


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SESSIONS_HAVE_OWNER, reason="Sessions have no owner before HA 2026.11"
)
async def test_overlapping_turn_cannot_replace_active_session_owner(
    hass: HomeAssistant,
) -> None:
    """A competing turn cannot borrow a session while another speaker is using it."""
    ready = asyncio.Event()
    hold = asyncio.Event()
    with chat_session.async_get_chat_session(hass, None, context=Context()) as session:

        async def active_turn():
            with _scoped_session_owner(session.conversation_id, "alice"):
                ready.set()
                await hold.wait()

        task = asyncio.create_task(active_turn())
        try:
            await asyncio.wait_for(ready.wait(), 5)
            with _scoped_session_owner(session.conversation_id, "bob"):
                assert session.user_id == "alice"
                with pytest.raises(Unauthorized):
                    with chat_session.async_get_chat_session(
                        hass, session.conversation_id, context=Context(user_id="bob")
                    ):
                        pytest.fail("A competing user must not enter the session")
            assert session.user_id == "alice"
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert session.user_id is None

    # Cancellation must not pin the stored session to Alice on a later turn.
    with chat_session.async_get_chat_session(
        hass, session.conversation_id, context=Context()
    ) as resumed:
        with _scoped_session_owner(resumed.conversation_id, "bob"):
            with chat_session.async_get_chat_session(
                hass, resumed.conversation_id, context=Context(user_id="bob")
            ):
                assert resumed.user_id == "bob"
        assert resumed.user_id is None


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SESSIONS_HAVE_OWNER, reason="Sessions have no owner before HA 2026.11"
)
async def test_session_owner_scope_ignores_other_conversations(
    hass: HomeAssistant,
) -> None:
    """Enrichment must not alter an unrelated active session."""
    with chat_session.async_get_chat_session(hass, None, context=Context()) as session:
        with _scoped_session_owner("other-conversation", "alice"):
            assert session.user_id is None


class GetDateTimeTool(llm.Tool):
    """Advertise a clock tool so HA does not add a changing clock to the prompt."""

    name = "GetDateTime"
    integration = "conversation"

    async def async_call(self, hass, tool_input, llm_context):
        return {}


class StaticPromptAPI(llm.API):
    """Provide a static API prompt while still passing each turn's context."""

    async def async_get_api_instance(self, llm_context):
        return llm.APIInstance(
            api=self,
            api_prompt="Use tools for current information.",
            llm_context=llm_context,
            tools=[GetDateTimeTool()],
        )


class ChatLogRecordingSource(StreamingSource):
    """Source agent that renders a prompt and records what it was given."""

    def __init__(
        self, prompt: str = "Base prompt.", api: llm.API | None = None
    ) -> None:
        super().__init__()
        self.prompt = prompt
        self.api = api
        self.system_prompts: list[str] = []
        self.user_messages: list[list[str]] = []
        self.api_user_ids: list[str | None] = []

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        await chat_log.async_provide_llm_data(
            user_input.as_llm_context("test"), self.api, self.prompt, None
        )
        self.system_prompts.append(chat_log.content[0].content)
        if self.api is not None:
            self.api_user_ids.append(chat_log.llm_api.llm_context.context.user_id)
        self.user_messages.append(
            [content.content for content in chat_log.content if content.role == "user"]
        )
        return await super()._async_handle_message(user_input, chat_log)


NOTE_ON = {CONF_SPEAKER_NOTE_ENABLED: True}
QUESTION = "tell me something long"


@pytest.mark.asyncio
async def test_speaker_note_names_the_speaker_of_each_turn(
    hass: HomeAssistant, freezer
) -> None:
    """A static API and clock tool keep the complete prompt stable across speakers."""
    freezer.move_to("2026-11-06 12:00:00+00:00")
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    source = ChatLogRecordingSource(
        api=StaticPromptAPI(hass=hass, id="test", name="Test")
    )
    await _setup_proxy(hass, source, options=NOTE_ON)

    set_correlated_recognition(_recognition(alice.id))
    first = await _run_turn(hass, source)
    assert first.errors == []

    source.tts_started.clear()
    freezer.tick(2)
    second = await _run_turn(hass, source, first.conversation_id)
    assert second.errors == []

    source.tts_started.clear()
    freezer.tick(2)
    set_correlated_recognition(_recognition(bob.id))
    third = await _run_turn(hass, source, first.conversation_id)
    assert third.errors == []
    assert third.tts_start_streaming is True
    assert source.user_id_at_first_delta == bob.id

    # One message per turn, in order: earlier turns must never change.
    assert source.user_messages[-1] == [
        f"[Speaker: Alice] {QUESTION}",
        f"[Speaker: unknown] {QUESTION}",
        f"[Speaker: Bob] {QUESTION}",
    ]
    assert len(set(source.system_prompts)) == 1
    assert "Base prompt." in source.system_prompts[0]
    assert "Use tools for current information." in source.system_prompts[0]
    assert "Alice" not in source.system_prompts[0]
    assert "Bob" not in source.system_prompts[0]
    assert source.api_user_ids == [alice.id, None, bob.id]


@pytest.mark.asyncio
async def test_speaker_note_does_not_freeze_other_prompt_content(
    hass: HomeAssistant, freezer
) -> None:
    """Without a clock tool, HA's time prompt still changes even with speaker notes."""
    freezer.move_to("2026-11-06 12:00:00+00:00")
    alice = await hass.auth.async_create_user("Alice")
    source = ChatLogRecordingSource()
    await _setup_proxy(hass, source, options=NOTE_ON)
    set_correlated_recognition(_recognition(alice.id))
    first = await _run_turn(hass, source)
    assert first.errors == []

    freezer.tick(2)
    source.tts_started.clear()
    second = await _run_turn(hass, source, first.conversation_id)
    assert second.errors == []
    assert source.system_prompts[0] != source.system_prompts[1]
    assert all("Base prompt." in prompt for prompt in source.system_prompts)
    assert source.user_messages[-1] == [
        f"[Speaker: Alice] {QUESTION}",
        f"[Speaker: unknown] {QUESTION}",
    ]


@pytest.mark.asyncio
async def test_speaker_note_is_off_by_default(hass: HomeAssistant) -> None:
    """Existing proxies keep sending the sentence as it was said."""
    alice = await hass.auth.async_create_user("Alice")
    source = ChatLogRecordingSource()
    await _setup_proxy(hass, source)
    set_correlated_recognition(_recognition(alice.id))

    turn = await _run_turn(hass, source)

    assert turn.errors == []
    assert source.user_messages[-1] == [QUESTION]


@pytest.mark.asyncio
async def test_speaker_note_template_can_use_home_assistant_states(
    hass: HomeAssistant,
) -> None:
    """The template can read preferences of the current speaker."""
    alice = await hass.auth.async_create_user("Alice")
    hass.states.async_set("input_text.likes_alice", "Be brief")
    source = ChatLogRecordingSource()
    await _setup_proxy(
        hass,
        source,
        options={
            **NOTE_ON,
            CONF_SPEAKER_NOTE_TEMPLATE: (
                "[{{ user_name }}: "
                "{{ states('input_text.likes_' ~ (user_name | slugify)) }}]"
            ),
        },
    )
    set_correlated_recognition(_recognition(alice.id))

    await _run_turn(hass, source)

    assert source.user_messages[-1] == [f"[Alice: Be brief] {QUESTION}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template",
    ["{{ 1 / 0 }}", "{{ '' }}"],
    ids=["broken_template", "empty_note"],
)
async def test_speaker_note_never_costs_the_turn(
    hass: HomeAssistant, template: str
) -> None:
    """A broken or empty note leaves the sentence as it was said."""
    source = ChatLogRecordingSource()
    await _setup_proxy(
        hass,
        source,
        options={**NOTE_ON, CONF_SPEAKER_NOTE_TEMPLATE: template},
    )

    turn = await _run_turn(hass, source)

    assert turn.errors == []
    assert source.user_messages[-1] == [QUESTION]


@pytest.mark.asyncio
async def test_speaker_note_options_are_offered_and_saved(hass: HomeAssistant) -> None:
    """The real options flow shows the fields and stores them."""
    proxy = await _setup_proxy(hass, NonStreamingSource())
    entry = proxy._config_entry
    entry.mock_state(hass, ConfigEntryState.LOADED)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["step_id"] == "conversation_options"
    fields = {str(key) for key in result["data_schema"].schema}
    assert {CONF_SPEAKER_NOTE_ENABLED, CONF_SPEAKER_NOTE_TEMPLATE} <= fields

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_CONVERSATION_ENTITY: SOURCE_ENTITY_ID,
            CONF_MIN_CONFIDENCE: 0.75,
            CONF_SPEAKER_NOTE_ENABLED: True,
            CONF_SPEAKER_NOTE_TEMPLATE: "[{{ user_name }}]",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_SPEAKER_NOTE_ENABLED] is True
    assert entry.options[CONF_SPEAKER_NOTE_TEMPLATE] == "[{{ user_name }}]"
