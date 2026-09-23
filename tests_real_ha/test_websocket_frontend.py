"""Real Home Assistant acceptance tests for WebSocket and frontend resources."""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.components import frontend
from homeassistant.core import HomeAssistant
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
from custom_components.speaker_recognition.frontend import PANEL_URL_PATH


async def _setup_integration(hass: HomeAssistant) -> None:
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()


def _entry(
    hass: HomeAssistant,
    *,
    entry_type: str,
    title: str,
    data: dict[str, Any],
    options: dict[str, Any] | None = None,
    unique_id: str | None = None,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title=title,
        data={CONF_ENTRY_TYPE: entry_type, **data},
        options=options or {},
        unique_id=unique_id,
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_frontend_panel_is_registered_admin_only(hass: HomeAssistant) -> None:
    """Integration setup registers exactly one admin-only frontend panel."""
    await _setup_integration(hass)

    assert frontend.async_panel_exists(hass, PANEL_URL_PATH)
    panel = hass.data[frontend.DATA_PANELS][PANEL_URL_PATH]
    assert panel.require_admin is True
    assert panel.config_panel_domain == DOMAIN
    assert panel.sidebar_title == "Speaker Recognition"


@pytest.mark.asyncio
async def test_repeated_component_setup_does_not_duplicate_frontend_resources(
    hass: HomeAssistant,
) -> None:
    """HA-level repeated setup remains idempotent for the panel/resources."""
    await _setup_integration(hass)
    first_panel = hass.data[frontend.DATA_PANELS][PANEL_URL_PATH]

    await _setup_integration(hass)

    panels = hass.data[frontend.DATA_PANELS]
    assert panels[PANEL_URL_PATH] is first_panel
    assert sum(path == PANEL_URL_PATH for path in panels) == 1


@pytest.mark.asyncio
async def test_status_websocket_command_runs_through_real_ha_connection(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """The registered status command is reachable through HA's WebSocket API."""
    user = await hass.auth.async_create_user("Alice")
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id({"type": f"{DOMAIN}/status"})
    response = await client.receive_json()

    assert response["success"] is True
    result = response["result"]
    assert any(item["id"] == user.id for item in result["users"])
    assert result["configured"] is False
    assert result["minimum_samples"] == 5
    assert result["microphone_secure_context_required"] is True


@pytest.mark.asyncio
async def test_websocket_commands_require_admin(
    hass: HomeAssistant,
    hass_ws_client: Any,
    hass_read_only_access_token: str,
) -> None:
    """A normal HA user cannot use Speaker Recognition management commands."""
    await _setup_integration(hass)
    client = await hass_ws_client(hass, hass_read_only_access_token)

    await client.send_json_auto_id({"type": f"{DOMAIN}/status"})
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
async def test_settings_websocket_returns_effective_entries_without_secret(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """Settings serialize real config entries but never expose the backend token."""
    main = _entry(
        hass,
        entry_type=ENTRY_TYPE_MAIN,
        title="Speaker Recognition",
        data={CONF_BACKEND_URL: "http://127.0.0.1:8099", CONF_BACKEND_TOKEN: "secret"},
        options={CONF_VOICE_SAMPLES: []},
        unique_id=ENTRY_TYPE_MAIN,
    )
    stt_entry = _entry(
        hass,
        entry_type=ENTRY_TYPE_STT,
        title="STT proxy",
        data={CONF_STT_ENTITY: "stt.source", CONF_USE_BASIC_DSP: False},
        options={CONF_USE_BASIC_DSP: True},
    )
    conversation_entry = _entry(
        hass,
        entry_type=ENTRY_TYPE_CONVERSATION,
        title="Conversation proxy",
        data={CONF_CONVERSATION_ENTITY: "conversation.source", CONF_MIN_CONFIDENCE: 0.7},
    )
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id({"type": f"{DOMAIN}/settings"})
    response = await client.receive_json()

    assert response["success"] is True
    result = response["result"]
    assert result["main"] == {
        "entry_id": main.entry_id,
        "title": "Speaker Recognition",
        "backend_url": "http://127.0.0.1:8099",
        "acceptance_thresholds": {"min_similarity": 0.55, "min_margin": 0.05},
    }
    assert CONF_BACKEND_TOKEN not in str(result)
    assert result["stt_entries"] == [
        {
            "entry_id": stt_entry.entry_id,
            "title": "STT proxy",
            "stt_entity": "stt.source",
            "stt_options": ["stt.source"],
            "use_basic_dsp": True,
        }
    ]
    assert result["conversation_entries"] == [
        {
            "entry_id": conversation_entry.entry_id,
            "title": "Conversation proxy",
            "conversation_entity": "conversation.source",
            "conversation_options": ["conversation.source"],
            "min_confidence": 0.7,
        }
    ]


@pytest.mark.asyncio
async def test_update_settings_unknown_entry_returns_controlled_error(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """Frontend settings reject stale config-entry IDs cleanly."""
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id(
        {
            "type": f"{DOMAIN}/update_settings",
            "entry_id": "missing-entry",
            "backend_url": "http://127.0.0.1:9000",
        }
    )
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == "unknown_entry"


@pytest.mark.asyncio
async def test_update_main_settings_rejects_empty_backend_url(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """The WebSocket schema/handler boundary rejects an unusable backend URL."""
    main = _entry(
        hass,
        entry_type=ENTRY_TYPE_MAIN,
        title="Speaker Recognition",
        data={CONF_BACKEND_URL: "http://127.0.0.1:8099", CONF_BACKEND_TOKEN: ""},
        options={CONF_VOICE_SAMPLES: []},
        unique_id=ENTRY_TYPE_MAIN,
    )
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id(
        {
            "type": f"{DOMAIN}/update_settings",
            "entry_id": main.entry_id,
            "backend_url": "   ",
        }
    )
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == "invalid_backend_url"


@pytest.mark.asyncio
async def test_stage_sample_invalid_payload_returns_websocket_error(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """Malformed browser audio is rejected through the real WebSocket path."""
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id(
        {
            "type": f"{DOMAIN}/stage_sample",
            "user_id": "user",
            "sample_index": 0,
            "wav_base64": "not-base64!",
        }
    )
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == "invalid_sample"


@pytest.mark.asyncio
async def test_websocket_schema_rejects_missing_required_fields(
    hass: HomeAssistant, hass_ws_client: Any
) -> None:
    """HA's registered command schema rejects malformed messages before execution."""
    await _setup_integration(hass)
    client = await hass_ws_client(hass)

    await client.send_json_auto_id({"type": f"{DOMAIN}/stage_sample"})
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == "invalid_format"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_error"),
    [
        ("commit_enrollment", "not_configured"),
        ("test_sample", "not_ready"),
    ],
)
async def test_commands_without_main_entry_fail_cleanly(
    hass: HomeAssistant,
    hass_ws_client: Any,
    command: str,
    expected_error: str,
) -> None:
    """Commands requiring the main integration report controlled missing-state errors."""
    await _setup_integration(hass)
    client = await hass_ws_client(hass)
    payload: dict[str, Any] = {"type": f"{DOMAIN}/{command}"}
    if command == "commit_enrollment":
        payload["user_id"] = "user"
    else:
        payload["wav_base64"] = "AA=="

    await client.send_json_auto_id(payload)
    response = await client.receive_json()

    assert response["success"] is False
    assert response["error"]["code"] == expected_error

@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-0.01, 1.01, "nan", "inf"])
async def test_backend_threshold_settings_reject_invalid_values(hass, hass_ws_client, value):
    main = _entry(hass, entry_type=ENTRY_TYPE_MAIN, title="Main", data={CONF_BACKEND_URL: "http://localhost:8099"})
    await _setup_integration(hass)
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/update_settings", "entry_id": main.entry_id,
        "backend_url": "http://localhost:8099",
        "acceptance_thresholds": {"min_similarity": value, "min_margin": 0.05},
    })
    assert not (await client.receive_json())["success"]
    assert "acceptance_thresholds" not in main.options


@pytest.mark.asyncio
async def test_backend_threshold_settings_require_admin(hass, hass_ws_client, hass_read_only_access_token):
    main = _entry(hass, entry_type=ENTRY_TYPE_MAIN, title="Main", data={CONF_BACKEND_URL: "http://localhost:8099"})
    await _setup_integration(hass)
    client = await hass_ws_client(hass, hass_read_only_access_token)
    await client.send_json_auto_id({
        "type": f"{DOMAIN}/update_settings", "entry_id": main.entry_id,
        "backend_url": "http://localhost:8099",
        "acceptance_thresholds": {"min_similarity": 0.7, "min_margin": 0.0},
    })
    response = await client.receive_json()
    assert response["error"]["code"] == "unauthorized"
    assert "acceptance_thresholds" not in main.options
