"""Acceptance tests bridging real Home Assistant to the real add-on container."""

from __future__ import annotations

import os
from pathlib import Path
import wave
from unittest.mock import patch

import pytest
from homeassistant.components import media_source
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_SAMPLES,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)

REAL_ADDON_URL = os.environ.get("SPEAKER_RECOGNITION_REAL_ADDON_URL")
REAL_ADDON_TOKEN = os.environ.get("SPEAKER_RECOGNITION_REAL_ADDON_TOKEN", "")
REAL_ADDON_AUDIO_DIR = os.environ.get("SPEAKER_RECOGNITION_REAL_ADDON_AUDIO_DIR")

pytestmark = pytest.mark.skipif(
    not REAL_ADDON_URL or not REAL_ADDON_AUDIO_DIR,
    reason="real add-on acceptance environment is not configured",
)


def _audio_dir() -> Path:
    assert REAL_ADDON_AUDIO_DIR is not None
    return Path(REAL_ADDON_AUDIO_DIR)


def _media_item(name: str) -> dict[str, str]:
    return {
        "media_content_id": f"media-source://media_source/local/{name}",
        "media_content_type": "audio/wav",
    }


def _main_entry(
    user_id: str,
    *,
    token: str | None = None,
) -> MockConfigEntry:
    assert REAL_ADDON_URL is not None
    samples = [_media_item(f"train-{index}.wav") for index in range(1, 4)]
    return MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: REAL_ADDON_URL,
            CONF_BACKEND_TOKEN: REAL_ADDON_TOKEN if token is None else token,
        },
        options={
            CONF_VOICE_SAMPLES: [
                {
                    CONF_USER: user_id,
                    CONF_SAMPLES: samples,
                    "sample_metadata": [
                        {"phrase": f"real-addon-{index}"} for index in range(1, 4)
                    ],
                }
            ]
        },
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )


def _resolve_media(_hass: HomeAssistant, media_id: str, _target: object) -> media_source.PlayMedia:
    name = media_id.rsplit("/", 1)[-1]
    path = _audio_dir() / name
    assert path.is_file(), f"missing real add-on audio fixture: {path}"
    return media_source.PlayMedia(
        url=f"/media/{name}",
        mime_type="audio/wav",
        path=path,
    )


def _pcm_from_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        sample_rate = wav_file.getframerate()
        return wav_file.readframes(wav_file.getnframes()), sample_rate


@pytest.mark.asyncio
async def test_real_ha_main_entry_trains_recognizes_and_reconciles_real_addon(
    hass: HomeAssistant,
) -> None:
    """HA setup and recognition cross the genuine integration/add-on HTTP boundary."""
    user = await hass.auth.async_create_user("Real Add-on Alice")
    entry = _main_entry(user.id)
    entry.add_to_hass(hass)

    with patch(
        "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
        side_effect=_resolve_media,
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)

    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.enrolled_users == {user.id}

    pcm, sample_rate = _pcm_from_wav(_audio_dir() / "recognize.wav")
    result = await entry.runtime_data.async_recognize(pcm, sample_rate=sample_rate)
    assert result is not None
    assert result.accepted is True
    assert result.user_id == user.id
    assert result.candidate_user_id == user.id
    assert result.engine_id == "resemblyzer"
    assert result.confidence > 0.0

    # A persisted HA policy is applied and acknowledged by the actual backend,
    # including after the HA runtime is replaced. No extra inference is needed.
    policy = {"min_similarity": 1.0, "min_margin": 0.0}
    hass.config_entries.async_update_entry(
        entry, options={**entry.options, "acceptance_thresholds": policy}
    )
    await hass.async_block_till_done(wait_background_tasks=True)
    strict = await entry.runtime_data.async_recognize(pcm, sample_rate=sample_rate)
    assert strict is not None
    assert strict.accepted == (strict.similarity >= 1.0)
    assert entry.runtime_data.acceptance_thresholds == policy
    health = await entry.runtime_data._async_get("/health")
    assert health["supports_acceptance_thresholds"] is True
    assert health["acceptance_thresholds"] == {"min_similarity": 0.55, "min_margin": 0.05}

    # Exercise HA's real update-listener path against the real add-on: removing the
    # configured user must delete the backend profile and reload a clean runtime.
    hass.config_entries.async_update_entry(entry, options={CONF_VOICE_SAMPLES: []})
    await hass.async_block_till_done(wait_background_tasks=True)

    assert entry.state is ConfigEntryState.LOADED
    assert entry.runtime_data.enrolled_users == set()
    assert entry.runtime_data.configured_users == set()


@pytest.mark.asyncio
async def test_real_ha_wrong_addon_token_fails_setup_closed(hass: HomeAssistant) -> None:
    """A real add-on 401 never leaves the HA main entry partially loaded."""
    user = await hass.auth.async_create_user("Wrong Token User")
    entry = _main_entry(user.id, token="definitely-wrong-token")
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_RETRY
