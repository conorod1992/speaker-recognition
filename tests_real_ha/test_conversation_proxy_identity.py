"""Real Home Assistant acceptance tests for Conversation identity propagation."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.components import conversation
from homeassistant.core import Context, HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.intent import IntentResponse, IntentResponseErrorCode
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
)


SOURCE_ENTITY_ID = "conversation.test_source"
MIN_CONFIDENCE = 0.75


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


def _conversation_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Conversation proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: SOURCE_ENTITY_ID,
            CONF_MIN_CONFIDENCE: MIN_CONFIDENCE,
        },
        unique_id=f"{ENTRY_TYPE_CONVERSATION}_{SOURCE_ENTITY_ID}",
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


def _proxy_entity(
    hass: HomeAssistant,
    main: MockConfigEntry,
    proxy_entry: MockConfigEntry | None = None,
) -> SpeakerRecognitionConversationEntity:
    proxy_entry = proxy_entry or _conversation_entry(hass)
    entity = SpeakerRecognitionConversationEntity(
        hass,
        proxy_entry.title,
        SOURCE_ENTITY_ID,
        proxy_entry.entry_id,
        proxy_entry,
        main,
    )
    entity.hass = hass
    return entity


def _input(
    *,
    user_id: str | None = None,
    conversation_id: str | None = "conversation-1",
) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text="turn on the kitchen light",
        context=Context(user_id=user_id),
        conversation_id=conversation_id,
        device_id="device-1",
        satellite_id="assist_satellite.kitchen",
        language="en",
        agent_id=SOURCE_ENTITY_ID,
        extra_system_prompt="keep this prompt",
    )


def _recognition(
    user_id: str,
    *,
    confidence: float = 0.94,
    accepted: bool = True,
    sequence: int = 1,
) -> CorrelatedRecognition:
    return CorrelatedRecognition(
        user_id=user_id,
        candidate_user_id=user_id,
        confidence=confidence,
        similarity=0.88,
        margin=0.31,
        accepted=accepted,
        all_scores={user_id: 0.88},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=sequence,
    )


class FakeSourceAgent:
    """Minimal external Conversation provider boundary."""

    supported_languages = ["en"]

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.received: conversation.ConversationInput | None = None

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        self.received = user_input
        if self.fail:
            raise RuntimeError("source conversation failed")
        response = IntentResponse(language=user_input.language)
        response.async_set_speech("source response")
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
            continue_conversation=True,
        )


async def _open_conversation_flow(hass: HomeAssistant) -> dict:
    _main_entry(hass)
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "menu"
    return await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_conversation"}
    )


@pytest.fixture(autouse=True)
def _clear_turn_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_real_config_flow_creates_conversation_proxy_entry(
    hass: HomeAssistant,
) -> None:
    """The genuine HA flow creates a current-version Conversation proxy."""
    result = await _open_conversation_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "add_conversation"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CONVERSATION_ENTITY: SOURCE_ENTITY_ID,
            CONF_MIN_CONFIDENCE: MIN_CONFIDENCE,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    entry = result["result"]
    assert entry.version == 2
    assert entry.data[CONF_ENTRY_TYPE] == ENTRY_TYPE_CONVERSATION
    assert entry.data[CONF_CONVERSATION_ENTITY] == SOURCE_ENTITY_ID
    assert entry.data[CONF_MIN_CONFIDENCE] == MIN_CONFIDENCE
    assert entry.unique_id == f"{ENTRY_TYPE_CONVERSATION}_{SOURCE_ENTITY_ID}"


@pytest.mark.asyncio
async def test_conversation_flow_rejects_duplicate_wrapped_source(
    hass: HomeAssistant,
) -> None:
    """One Conversation source cannot be wrapped twice."""
    _main_entry(hass)
    _conversation_entry(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_conversation"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CONVERSATION_ENTITY: SOURCE_ENTITY_ID,
            CONF_MIN_CONFIDENCE: MIN_CONFIDENCE,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "proxy_source_already_wrapped"}


@pytest.mark.asyncio
async def test_conversation_flow_rejects_speaker_recognition_proxy_source(
    hass: HomeAssistant,
) -> None:
    """Recursive Speaker Recognition Conversation proxy chains are rejected."""
    _main_entry(hass)
    wrapped_entry = MockConfigEntry(
        domain=DOMAIN,
        title="Existing conversation proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: "conversation.original",
            CONF_MIN_CONFIDENCE: MIN_CONFIDENCE,
        },
        unique_id=f"{ENTRY_TYPE_CONVERSATION}_conversation.original",
        version=2,
        minor_version=0,
    )
    wrapped_entry.add_to_hass(hass)
    recursive_source = er.async_get(hass).async_get_or_create(
        "conversation",
        DOMAIN,
        "existing-proxy",
        suggested_object_id="speaker_recognition_proxy",
        config_entry=wrapped_entry,
    ).entity_id

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "add_conversation"}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_CONVERSATION_ENTITY: recursive_source,
            CONF_MIN_CONFIDENCE: MIN_CONFIDENCE,
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "speaker_recognition_proxy_source"}


@pytest.mark.asyncio
async def test_recognized_user_enriches_anonymous_conversation_context(
    hass: HomeAssistant,
) -> None:
    """A high-confidence recognized HA user becomes the downstream context user."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    source = FakeSourceAgent()
    set_correlated_recognition(_recognition(alice.id))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        result = await entity.async_process(_input())

    assert source.received is not None
    assert source.received.context.user_id == alice.id
    assert source.received.text == "turn on the kitchen light"
    assert source.received.device_id == "device-1"
    assert source.received.satellite_id == "assist_satellite.kitchen"
    assert source.received.extra_system_prompt == "keep this prompt"
    assert result.conversation_id == "conversation-1"
    assert result.continue_conversation is True


@pytest.mark.asyncio
async def test_existing_authenticated_context_wins_over_recognized_speaker(
    hass: HomeAssistant,
) -> None:
    """Speaker recognition cannot replace a user already authenticated by HA."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    source = FakeSourceAgent()
    set_correlated_recognition(_recognition(alice.id))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        await entity.async_process(_input(user_id=bob.id))

    assert source.received is not None
    assert source.received.context.user_id == bob.id


@pytest.mark.asyncio
async def test_below_threshold_recognition_does_not_create_identity(
    hass: HomeAssistant,
) -> None:
    """Accepted backend candidates below the configured threshold fail closed."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    source = FakeSourceAgent()
    set_correlated_recognition(_recognition(alice.id, confidence=0.70))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        await entity.async_process(_input())

    assert source.received is not None
    assert source.received.context.user_id is None


@pytest.mark.asyncio
async def test_rejected_recognition_does_not_create_identity(
    hass: HomeAssistant,
) -> None:
    """A non-accepted recognition result cannot become HA identity context."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    source = FakeSourceAgent()
    set_correlated_recognition(_recognition(alice.id, accepted=False))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        await entity.async_process(_input())

    assert source.received is not None
    assert source.received.context.user_id is None


@pytest.mark.asyncio
async def test_deleted_ha_user_cannot_be_reintroduced_by_stale_recognition(
    hass: HomeAssistant,
) -> None:
    """A stale backend user ID is rejected after that HA user is deleted."""
    await hass.auth.async_create_user("Owner")
    alice = await hass.auth.async_create_user("Alice")
    assert not alice.is_owner
    stale_user_id = alice.id
    await hass.auth.async_remove_user(alice)
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    source = FakeSourceAgent()
    set_correlated_recognition(_recognition(stale_user_id))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        await entity.async_process(_input())

    assert source.received is not None
    assert source.received.context.user_id is None


@pytest.mark.asyncio
async def test_missing_source_conversation_returns_controlled_error(
    hass: HomeAssistant,
) -> None:
    """A disappeared source agent returns a model-visible HA conversation error."""
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=None,
    ):
        result = await entity.async_process(_input())

    assert result.response.error_code is IntentResponseErrorCode.FAILED_TO_HANDLE
    assert result.conversation_id is None


@pytest.mark.asyncio
async def test_source_conversation_failure_propagates_after_identity_consumption(
    hass: HomeAssistant,
) -> None:
    """Underlying agent failures propagate without letting identity leak to a later turn."""
    alice = await hass.auth.async_create_user("Alice")
    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    set_correlated_recognition(_recognition(alice.id))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=FakeSourceAgent(fail=True),
    ):
        with pytest.raises(RuntimeError, match="source conversation failed"):
            await entity.async_process(_input())

    clean_source = FakeSourceAgent()
    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=clean_source,
    ):
        await entity.async_process(_input(conversation_id="conversation-2"))

    assert clean_source.received is not None
    assert clean_source.received.context.user_id is None


@pytest.mark.asyncio
async def test_captured_enrollment_turn_never_reaches_source_agent(
    hass: HomeAssistant,
) -> None:
    """A spoken enrollment phrase must be consumed even when it sounds like a command."""
    from custom_components.speaker_recognition.enrollment import (
        consume_completed_enrollment_turn,
    )

    main = _main_entry(hass)
    entity = _proxy_entity(hass, main)
    agent = FakeSourceAgent()
    sequence = 987654
    hass.data.setdefault(DOMAIN, {})["completed_enrollment_turns"] = {sequence}
    set_correlated_recognition(_recognition("enrolled-user", sequence=sequence))

    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=agent,
    ):
        response = await entity.async_process(_input())

    assert agent.received is None
    assert response.conversation_id == "conversation-1"
    assert not consume_completed_enrollment_turn(hass, sequence)


@pytest.mark.asyncio
async def test_discard_waits_for_inflight_capture(
    hass: HomeAssistant, tmp_path, monkeypatch
) -> None:
    """Discard must not succeed before a concurrent sample has been staged."""
    import asyncio
    from custom_components.speaker_recognition import enrollment

    hass.config.media_dirs["local"] = str(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    original = hass.async_add_executor_job

    async def delayed_executor(target, *args):
        if target is enrollment._write_wav:
            entered.set()
            await release.wait()
        return await original(target, *args)

    monkeypatch.setattr(hass, "async_add_executor_job", delayed_executor)
    capture = asyncio.create_task(enrollment.async_stage_pcm_sample(
        hass, "capture-test", 0, bytes([1, 0]) * 8000, 16000
    ))
    await asyncio.wait_for(entered.wait(), 5)
    discard = asyncio.create_task(
        enrollment.async_discard_staged_samples(hass, "capture-test")
    )
    await asyncio.sleep(0)
    assert not discard.done()
    release.set()
    await capture
    assert await discard == 1
    assert not enrollment.staged_samples(hass, "capture-test")
