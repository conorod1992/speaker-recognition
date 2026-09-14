"""Real Home Assistant acceptance tests for config flow and guided enrollment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import wave

import pytest
from homeassistant import config_entries
from homeassistant.components import media_source
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.config_flow import ENROLLMENT_PHRASES
from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_FINISH_ENROLLMENT,
    CONF_PENDING_ENROLLMENT,
    CONF_SAMPLE,
    CONF_SAMPLES,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)

BACKEND_URL = "http://127.0.0.1:64123"


def _write_wav(path: Path, *, frames: int = 16000) -> None:
    """Write a small valid mono 16-bit PCM WAV fixture."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * frames)


def _media_item(name: str) -> dict[str, str]:
    return {
        "media_content_id": f"media-source://media_source/local/{name}",
        "media_content_type": "audio/wav",
    }


async def _start_main_flow(hass: HomeAssistant) -> dict:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "main"
    return result


async def _start_enrollment_flow(hass: HomeAssistant) -> dict:
    result = await _start_main_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_BACKEND_URL: BACKEND_URL, CONF_BACKEND_TOKEN: "test-token"},
    )
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "enrollment_menu"
    assert set(result["menu_options"]) == {"enrollment_user", "finish_setup"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "enrollment_user"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_user"
    return result


async def _select_user(hass: HomeAssistant, result: dict, user_id: str) -> dict:
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_USER: user_id}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_sample"
    return result


@pytest.mark.asyncio
async def test_real_config_flow_reaches_enrollment_menu(hass: HomeAssistant) -> None:
    """The genuine HA config-flow manager runs the integration's main flow."""
    result = await _start_main_flow(hass)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_BACKEND_URL: BACKEND_URL, CONF_BACKEND_TOKEN: "secret"},
    )

    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "enrollment_menu"
    assert set(result["menu_options"]) == {"enrollment_user", "finish_setup"}


@pytest.mark.asyncio
async def test_existing_main_entry_routes_new_flow_to_proxy_menu(
    hass: HomeAssistant,
) -> None:
    """A second user flow cannot create another main entry."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: BACKEND_URL,
            CONF_BACKEND_TOKEN: "",
        },
        options={CONF_VOICE_SAMPLES: []},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "menu"
    assert set(result["menu_options"]) == {"add_stt", "add_conversation"}


@pytest.mark.asyncio
async def test_guided_enrollment_accepts_real_ha_user(hass: HomeAssistant) -> None:
    """A user from HA's real auth store can be selected for enrollment."""
    user = await hass.auth.async_create_user("Alice")
    result = await _start_enrollment_flow(hass)

    result = await _select_user(hass, result, user.id)

    assert result["description_placeholders"]["sample_number"] == "1"
    assert result["description_placeholders"]["accepted"] == "0"
    assert result["description_placeholders"]["phrase"] == ENROLLMENT_PHRASES[0]


@pytest.mark.asyncio
async def test_enrollment_rejects_non_media_source_selection(
    hass: HomeAssistant,
) -> None:
    """Enrollment rejects values that did not come from HA Media Source."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            CONF_SAMPLE: {
                "media_content_id": "https://example.invalid/alice.wav",
                "media_content_type": "audio/wav",
            }
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_sample"
    assert result["errors"] == {CONF_SAMPLE: "invalid_enrollment_sample"}


@pytest.mark.asyncio
async def test_enrollment_rejects_media_without_local_path(
    hass: HomeAssistant,
) -> None:
    """A resolvable Media Source item must still map to a local HA file."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="https://example.invalid/alice.wav",
            mime_type="audio/wav",
            path=None,
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_SAMPLE: _media_item("alice.wav")}
        )

    assert result["errors"] == {CONF_SAMPLE: "invalid_enrollment_sample"}


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_kind", ["malformed", "short"])
async def test_enrollment_rejects_invalid_local_wav(
    hass: HomeAssistant, tmp_path: Path, fixture_kind: str
) -> None:
    """Actual WAV decoding rejects malformed and implausibly short local audio."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)
    path = tmp_path / f"{fixture_kind}.wav"
    if fixture_kind == "malformed":
        path.write_bytes(b"not a wav")
    else:
        _write_wav(path, frames=1000)

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url=f"/media/{path.name}", mime_type="audio/wav", path=path
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_SAMPLE: _media_item(path.name)}
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_sample"
    assert result["errors"] == {CONF_SAMPLE: "invalid_enrollment_sample"}


@pytest.mark.asyncio
async def test_five_valid_samples_can_finish_enrollment_early(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Five genuine decoded samples satisfy the integration's minimum enrollment."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)
    path = tmp_path / "alice.wav"
    _write_wav(path)

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="/media/alice.wav", mime_type="audio/wav", path=path
        ),
    ):
        for index in range(5):
            user_input = {CONF_SAMPLE: _media_item(f"alice-{index}.wav")}
            if index == 4:
                user_input[CONF_FINISH_ENROLLMENT] = True
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], user_input
            )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_review"
    assert result["description_placeholders"]["accepted"] == "5"


@pytest.mark.asyncio
async def test_six_valid_samples_advance_to_review_automatically(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Collecting all prompted samples advances to review without an early finish."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)
    path = tmp_path / "alice.wav"
    _write_wav(path)

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="/media/alice.wav", mime_type="audio/wav", path=path
        ),
    ):
        for index in range(6):
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], {CONF_SAMPLE: _media_item(f"alice-{index}.wav")}
            )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_review"
    assert result["description_placeholders"]["accepted"] == "6"


@pytest.mark.asyncio
async def test_retry_replaces_only_selected_sample_and_persists_pending_enrollment(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Retrying one sample preserves the rest and creates pending enrollment data."""
    user = await hass.auth.async_create_user("Alice")
    result = await _select_user(hass, await _start_enrollment_flow(hass), user.id)
    original_path = tmp_path / "original.wav"
    replacement_path = tmp_path / "replacement.wav"
    _write_wav(original_path)
    _write_wav(replacement_path)

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="/media/original.wav", mime_type="audio/wav", path=original_path
        ),
    ):
        for index in range(5):
            user_input = {CONF_SAMPLE: _media_item(f"original-{index}.wav")}
            if index == 4:
                user_input[CONF_FINISH_ENROLLMENT] = True
            result = await hass.config_entries.flow.async_configure(
                result["flow_id"], user_input
            )

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"enrollment_action": "retry_1"}
    )
    assert result["step_id"] == "enrollment_sample"

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url="/media/replacement.wav", mime_type="audio/wav", path=replacement_path
        ),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {CONF_SAMPLE: _media_item("replacement.wav")}
        )

    assert result["step_id"] == "enrollment_review"
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"enrollment_action": "finish"}
    )
    assert result["step_id"] == "enrollment_complete"
    result = await hass.config_entries.flow.async_configure(result["flow_id"], {})

    assert result["type"] is FlowResultType.CREATE_ENTRY
    entry = result["result"]
    enrollment = entry.options[CONF_VOICE_SAMPLES][0]
    assert enrollment[CONF_USER] == user.id
    assert enrollment[CONF_SAMPLES] == [
        _media_item("original-0.wav"),
        _media_item("replacement.wav"),
        _media_item("original-2.wav"),
        _media_item("original-3.wav"),
        _media_item("original-4.wav"),
    ]
    assert entry.options[CONF_PENDING_ENROLLMENT] == user.id


@pytest.mark.asyncio
async def test_finish_setup_without_enrollment_creates_current_main_entry(
    hass: HomeAssistant,
) -> None:
    """Initial setup can intentionally finish with no enrolled speakers."""
    result = await _start_main_flow(hass)
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_BACKEND_URL: BACKEND_URL, CONF_BACKEND_TOKEN: "secret"},
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {"next_step_id": "finish_setup"}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    entry = result["result"]
    assert entry.version == 2
    assert entry.minor_version == 0
    assert entry.unique_id == ENTRY_TYPE_MAIN
    assert entry.data == {
        CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
        CONF_BACKEND_URL: BACKEND_URL,
        CONF_BACKEND_TOKEN: "secret",
    }
    assert entry.options[CONF_VOICE_SAMPLES] == []
