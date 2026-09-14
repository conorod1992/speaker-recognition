"""Real Home Assistant acceptance tests for events, history and diagnostics."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)
from custom_components.speaker_recognition.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.speaker_recognition.telemetry import get_decision_history


EVENT_TYPE = "speaker_recognition_detected"


async def _setup_integration(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()


def _main_entry(
    hass: HomeAssistant,
    *,
    token: str = "super-secret-token",
    options: dict[str, Any] | None = None,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: token,
        },
        options={CONF_VOICE_SAMPLES: [], **(options or {})},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


def _event_data(sequence: int, *, entity_id: str = "stt.speaker_recognition") -> dict[str, Any]:
    return {
        "user_id": "alice",
        "candidate_user_id": "alice",
        "confidence": 0.94,
        "similarity": 0.88,
        "margin": 0.31,
        "accepted": True,
        "all_scores": {"alice": 0.88},
        "entity_id": entity_id,
        "utterance_sequence": sequence,
        "stt_seconds": 0.20,
        "recognition_seconds": 0.05,
        "preparation_seconds": 0.01,
        "added_latency_seconds": 0.03,
        "audio_seconds": 1.25,
    }


@pytest.mark.asyncio
async def test_real_bus_event_records_one_authoritative_decision(
    hass: HomeAssistant,
) -> None:
    """A genuine HA bus event creates one history record and duplicate turn events dedupe."""
    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None

    data = _event_data(1)
    hass.bus.async_fire(EVENT_TYPE, data)
    hass.bus.async_fire(EVENT_TYPE, data)
    await hass.async_block_till_done()

    records = history.recent(25)
    assert len(records) == 1
    assert records[0]["utterance_sequence"] == 1
    assert records[0]["stt_entity_id"] == "stt.speaker_recognition"
    assert records[0]["accepted"] is True


@pytest.mark.asyncio
async def test_calibration_exclusion_is_consumed_exactly_once(
    hass: HomeAssistant,
) -> None:
    """An excluded calibration turn is skipped once, then the sequence can be recorded normally."""
    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None
    hass.data[DOMAIN].setdefault("calibration_excluded_utterances", set()).add(7)

    hass.bus.async_fire(EVENT_TYPE, _event_data(7))
    await hass.async_block_till_done()
    assert history.recent(25) == []
    assert 7 not in hass.data[DOMAIN]["calibration_excluded_utterances"]

    hass.bus.async_fire(EVENT_TYPE, _event_data(7))
    await hass.async_block_till_done()
    assert [item["utterance_sequence"] for item in history.recent(25)] == [7]


@pytest.mark.asyncio
async def test_repeated_component_setup_does_not_duplicate_event_listener(
    hass: HomeAssistant,
) -> None:
    """Repeated HA setup cannot record a single later event more than once."""
    await _setup_integration(hass)
    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None

    hass.bus.async_fire(EVENT_TYPE, _event_data(10))
    await hass.async_block_till_done()

    assert [item["utterance_sequence"] for item in history.recent(25)] == [10]


@pytest.mark.asyncio
async def test_bus_event_history_preserves_newest_first_turn_order(
    hass: HomeAssistant,
) -> None:
    """Independent STT turns are persisted in order and exposed newest first."""
    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None

    hass.bus.async_fire(EVENT_TYPE, _event_data(20))
    hass.bus.async_fire(EVENT_TYPE, _event_data(21))
    await hass.async_block_till_done()

    assert [item["utterance_sequence"] for item in history.recent(25)] == [21, 20]


@pytest.mark.asyncio
async def test_decision_history_loads_persisted_records_on_real_ha_setup(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
) -> None:
    """Existing decision history is restored from HA storage during integration setup."""
    persisted = {
        "decision_id": "persisted-decision",
        "created_at": "2026-09-14T20:00:00+00:00",
        "utterance_sequence": 33,
        "stt_entity_id": "stt.persisted",
        "accepted": True,
    }
    hass_storage[f"{DOMAIN}.decision_history"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.decision_history",
        "data": {"records": [persisted]},
    }

    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None

    assert history.recent(25) == [persisted]


@pytest.mark.asyncio
async def test_malformed_persisted_history_is_filtered_without_breaking_setup(
    hass: HomeAssistant,
    hass_storage: dict[str, Any],
) -> None:
    """Malformed stored items are ignored while valid dictionary records survive."""
    valid = {"decision_id": "valid", "utterance_sequence": 44}
    hass_storage[f"{DOMAIN}.decision_history"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.decision_history",
        "data": {"records": [None, "bad", 123, valid]},
    }

    await _setup_integration(hass)
    history = get_decision_history(hass)
    assert history is not None

    assert history.recent(25) == [valid]


@pytest.mark.asyncio
async def test_config_entry_diagnostics_are_available_and_redact_tokens(
    hass: HomeAssistant,
) -> None:
    """HA config-entry diagnostics exist and never expose backend credentials."""
    secret = "backend-token-that-must-not-leak"
    entry = _main_entry(hass, token=secret, options={CONF_BACKEND_TOKEN: secret})
    await _setup_integration(hass)

    hass.bus.async_fire(EVENT_TYPE, _event_data(50))
    await hass.async_block_till_done()
    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["entry"]["entry_id"] == entry.entry_id
    assert diagnostics["runtime"]["decision_history_count"] == 1
    assert diagnostics["entry"]["data"][CONF_BACKEND_TOKEN] != secret
    assert diagnostics["entry"]["options"][CONF_BACKEND_TOKEN] != secret
    assert secret not in str(diagnostics)


@pytest.mark.asyncio
async def test_config_entry_diagnostics_remain_useful_while_backend_is_offline(
    hass: HomeAssistant,
) -> None:
    """Diagnostics are generated from HA state without contacting the recognition backend."""
    entry = _main_entry(hass)
    await _setup_integration(hass)

    diagnostics = await async_get_config_entry_diagnostics(hass, entry)

    assert diagnostics["entry"]["state"] == entry.state.value
    assert diagnostics["runtime"]["loaded"] is False
    assert diagnostics["runtime"]["decision_history_count"] == 0
