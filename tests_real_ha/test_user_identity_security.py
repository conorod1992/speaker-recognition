"""Real Home Assistant acceptance tests for user identity security."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from homeassistant.components import conversation
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
)

SOURCE = "conversation.identity_security_source"
THRESHOLD = 0.75


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
        title="Identity security proxy",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION,
            CONF_CONVERSATION_ENTITY: SOURCE,
            CONF_MIN_CONFIDENCE: THRESHOLD,
        },
        unique_id=f"{ENTRY_TYPE_CONVERSATION}_{SOURCE}",
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


def _entity(hass: HomeAssistant) -> SpeakerRecognitionConversationEntity:
    main = _main_entry(hass)
    proxy = _conversation_entry(hass)
    entity = SpeakerRecognitionConversationEntity(
        hass,
        proxy.title,
        SOURCE,
        proxy.entry_id,
        proxy,
        main,
    )
    entity.hass = hass
    return entity


def _recognition(user_id: str, *, sequence: int = 1) -> CorrelatedRecognition:
    return CorrelatedRecognition(
        user_id=user_id,
        candidate_user_id=user_id,
        confidence=0.95,
        similarity=0.91,
        margin=0.32,
        accepted=True,
        all_scores={user_id: 0.91},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=sequence,
    )


def _input(*, user_id: str | None = None) -> conversation.ConversationInput:
    return conversation.ConversationInput(
        text="turn on the kitchen light",
        context=Context(user_id=user_id),
        conversation_id="identity-security",
        device_id="device-1",
        satellite_id="assist_satellite.kitchen",
        language="en",
        agent_id=SOURCE,
    )


class RecordingSourceAgent:
    """Minimal downstream Conversation provider boundary."""

    supported_languages = ["en"]

    def __init__(self) -> None:
        self.received: list[conversation.ConversationInput] = []

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        self.received.append(user_input)
        response = IntentResponse(language=user_input.language)
        response.async_set_speech("ok")
        return conversation.ConversationResult(
            response=response,
            conversation_id=user_input.conversation_id,
        )


async def _run(
    entity: SpeakerRecognitionConversationEntity,
    source: RecordingSourceAgent,
    user_input: conversation.ConversationInput,
) -> conversation.ConversationResult:
    with patch(
        "custom_components.speaker_recognition.conversation.conversation.async_get_agent",
        return_value=source,
    ):
        return await entity.async_process(user_input)


async def _create_non_owner_user(hass: HomeAssistant, name: str):
    if await hass.auth.async_get_owner() is None:
        await hass.auth.async_create_user("Owner")
    return await hass.auth.async_create_user(name)


@pytest.fixture(autouse=True)
def _clear_correlation() -> None:
    clear_correlated_recognition()
    yield
    clear_correlated_recognition()


@pytest.mark.asyncio
async def test_system_generated_user_cannot_be_injected_from_voice_recognition(
    hass: HomeAssistant,
) -> None:
    """Speaker recognition must never promote a HA system user into user context."""
    system_user = await hass.auth.async_create_system_user("Speaker service")
    entity = _entity(hass)
    source = RecordingSourceAgent()
    set_correlated_recognition(_recognition(system_user.id))

    await _run(entity, source, _input())

    assert source.received[-1].context.user_id is None


@pytest.mark.asyncio
async def test_inactive_user_cannot_be_injected_from_voice_recognition(
    hass: HomeAssistant,
) -> None:
    """A disabled HA account fails closed even if the backend still recognizes it."""
    user = await _create_non_owner_user(hass, "Alice")
    await hass.auth.async_update_user(user, is_active=False)
    entity = _entity(hass)
    source = RecordingSourceAgent()
    set_correlated_recognition(_recognition(user.id))

    await _run(entity, source, _input())

    assert source.received[-1].context.user_id is None


@pytest.mark.asyncio
async def test_reactivated_user_is_eligible_on_the_next_turn(
    hass: HomeAssistant,
) -> None:
    """Identity eligibility follows current HA account state without stale caching."""
    user = await _create_non_owner_user(hass, "Alice")
    entity = _entity(hass)
    source = RecordingSourceAgent()

    await hass.auth.async_update_user(user, is_active=False)
    set_correlated_recognition(_recognition(user.id, sequence=1))
    await _run(entity, source, _input())
    assert source.received[-1].context.user_id is None

    await hass.auth.async_update_user(user, is_active=True)
    set_correlated_recognition(_recognition(user.id, sequence=2))
    await _run(entity, source, _input())
    assert source.received[-1].context.user_id == user.id


@pytest.mark.asyncio
async def test_user_rename_preserves_identity_by_stable_ha_user_id(
    hass: HomeAssistant,
) -> None:
    """Renaming an enrolled HA user must not invalidate its stable identity mapping."""
    user = await hass.auth.async_create_user("Alice")
    stable_id = user.id
    await hass.auth.async_update_user(user, name="Alicia")
    entity = _entity(hass)
    source = RecordingSourceAgent()
    set_correlated_recognition(_recognition(stable_id))

    await _run(entity, source, _input())

    assert source.received[-1].context.user_id == stable_id
    renamed = await hass.auth.async_get_user(stable_id)
    assert renamed is not None and renamed.name == "Alicia"


@pytest.mark.asyncio
async def test_authenticated_context_still_wins_when_recognition_points_to_system_user(
    hass: HomeAssistant,
) -> None:
    """Recognition cannot replace an already authenticated HA user context."""
    authenticated = await hass.auth.async_create_user("Authenticated")
    system_user = await hass.auth.async_create_system_user("Speaker service")
    entity = _entity(hass)
    source = RecordingSourceAgent()
    set_correlated_recognition(_recognition(system_user.id))

    await _run(entity, source, _input(user_id=authenticated.id))

    assert source.received[-1].context.user_id == authenticated.id


@pytest.mark.asyncio
async def test_existing_system_context_is_preserved_but_not_created_by_recognition(
    hass: HomeAssistant,
) -> None:
    """The hardening applies to voice-derived identity, not HA's existing Context."""
    system_context = await hass.auth.async_create_system_user("Automation service")
    recognized = await hass.auth.async_create_user("Alice")
    entity = _entity(hass)
    source = RecordingSourceAgent()
    set_correlated_recognition(_recognition(recognized.id))

    await _run(entity, source, _input(user_id=system_context.id))

    assert source.received[-1].context.user_id == system_context.id
