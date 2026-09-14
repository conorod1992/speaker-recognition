"""Real Home Assistant acceptance tests for migration and media hardening."""

from __future__ import annotations

from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition import async_migrate_entry
from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_CONVERSATION_ENTITY,
    CONF_ENTRY_TYPE,
    CONF_STT_ENTITY,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_CONVERSATION,
    ENTRY_TYPE_MAIN,
    ENTRY_TYPE_STT,
)
from custom_components.speaker_recognition.enrollment import (
    _managed_media_path,
    async_cleanup_managed_samples,
    async_stage_pcm_sample,
)


def _legacy_entry(
    hass: HomeAssistant,
    *,
    data: dict,
    options: dict | None = None,
    unique_id: str | None = None,
    version: int = 1,
) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Legacy Speaker Recognition",
        data=data,
        options=options or {},
        unique_id=unique_id,
        version=version,
    )
    entry.add_to_hass(hass)
    return entry


@pytest.mark.asyncio
async def test_v1_main_migrates_without_losing_enrollment_or_backend_settings(
    hass: HomeAssistant,
) -> None:
    """Legacy main state survives the v1 -> v2 schema migration intact."""
    samples = [{"user": "alice", "samples": [{"media_content_id": "sample.wav"}]}]
    entry = _legacy_entry(
        hass,
        data={
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: "secret",
        },
        options={CONF_VOICE_SAMPLES: samples, "future_option": "preserve-me"},
    )

    assert await async_migrate_entry(hass, entry) is True

    assert entry.version == 2
    assert entry.minor_version == 0
    assert entry.data == {
        CONF_BACKEND_URL: "http://127.0.0.1:8099",
        CONF_BACKEND_TOKEN: "secret",
        CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
    }
    assert entry.options[CONF_VOICE_SAMPLES] == samples
    assert entry.options["future_option"] == "preserve-me"
    assert entry.unique_id == ENTRY_TYPE_MAIN


@pytest.mark.asyncio
async def test_v1_stt_proxy_infers_type_and_current_unique_id(hass: HomeAssistant) -> None:
    """Legacy STT proxies gain the v2 discriminator and stable source-based ID."""
    entry = _legacy_entry(hass, data={CONF_STT_ENTITY: "stt.legacy_source"})

    assert await async_migrate_entry(hass, entry) is True

    assert entry.data[CONF_ENTRY_TYPE] == ENTRY_TYPE_STT
    assert entry.data[CONF_STT_ENTITY] == "stt.legacy_source"
    assert entry.unique_id == "stt_stt.legacy_source"
    assert entry.version == 2


@pytest.mark.asyncio
async def test_v1_conversation_proxy_source_in_options_is_preserved(
    hass: HomeAssistant,
) -> None:
    """Migration honors effective proxy sources stored in legacy options."""
    entry = _legacy_entry(
        hass,
        data={},
        options={CONF_CONVERSATION_ENTITY: "conversation.legacy_source", "x": 1},
    )

    assert await async_migrate_entry(hass, entry) is True

    assert entry.data == {CONF_ENTRY_TYPE: ENTRY_TYPE_CONVERSATION}
    assert entry.options == {CONF_CONVERSATION_ENTITY: "conversation.legacy_source", "x": 1}
    assert entry.unique_id == "conversation_conversation.legacy_source"


@pytest.mark.asyncio
async def test_current_entry_migration_is_idempotent(hass: HomeAssistant) -> None:
    """Current entries pass migration without being rewritten."""
    entry = _legacy_entry(
        hass,
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: "",
        },
        options={CONF_VOICE_SAMPLES: [{"user": "alice", "samples": []}]},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
    )
    before_data = dict(entry.data)
    before_options = dict(entry.options)

    assert await async_migrate_entry(hass, entry) is True
    assert dict(entry.data) == before_data
    assert dict(entry.options) == before_options
    assert entry.unique_id == ENTRY_TYPE_MAIN


@pytest.mark.asyncio
async def test_unknown_future_entry_version_fails_closed(hass: HomeAssistant) -> None:
    """A schema newer than this integration is never silently downgraded."""
    entry = _legacy_entry(
        hass,
        data={CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN},
        unique_id=ENTRY_TYPE_MAIN,
        version=3,
    )

    assert await async_migrate_entry(hass, entry) is False
    assert entry.version == 3
    assert entry.data == {CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN}


def _set_media_root(hass: HomeAssistant, root: Path) -> None:
    hass.config.media_dirs.clear()
    hass.config.media_dirs["local"] = str(root)


def test_managed_media_resolver_rejects_traversal_and_unowned_media(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Cleanup resolution cannot escape the integration-owned media subtree."""
    _set_media_root(hass, tmp_path)

    valid_id = (
        "media-source://media_source/local/"
        "speaker_recognition_enrollment/alice/sample.wav"
    )
    traversal_id = (
        "media-source://media_source/local/"
        "speaker_recognition_enrollment/../outside.wav"
    )
    unrelated_id = "media-source://media_source/local/family/photo.wav"

    assert _managed_media_path(hass, valid_id) == (
        tmp_path / "speaker_recognition_enrollment/alice/sample.wav"
    ).resolve()
    assert _managed_media_path(hass, traversal_id) is None
    assert _managed_media_path(hass, unrelated_id) is None


@pytest.mark.asyncio
async def test_staged_sample_sanitizes_user_id_and_stays_inside_media_root(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """A hostile HA user ID cannot influence the managed filesystem path."""
    _set_media_root(hass, tmp_path)
    pcm = b"\x00\x00" * 8000

    result = await async_stage_pcm_sample(
        hass,
        "../../alice:unsafe/user",
        0,
        pcm,
        16000,
    )

    media_id = str(result["media_content_id"])
    assert ".." not in media_id
    path = _managed_media_path(hass, media_id)
    assert path is not None
    assert path.is_relative_to((tmp_path / "speaker_recognition_enrollment").resolve())
    assert path.exists()


@pytest.mark.asyncio
async def test_cleanup_ignores_missing_and_unmanaged_media_files(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Upgrade cleanup is idempotent and never deletes unrelated HA media."""
    _set_media_root(hass, tmp_path)
    unrelated = tmp_path / "family.wav"
    unrelated.write_bytes(b"keep")
    samples = [
        {
            "user": "alice",
            "samples": [
                {
                    "media_content_id": (
                        "media-source://media_source/local/"
                        "speaker_recognition_enrollment/alice/missing.wav"
                    )
                },
                {"media_content_id": "media-source://media_source/local/family.wav"},
            ],
        }
    ]

    await async_cleanup_managed_samples(hass, samples, {"alice"})

    assert unrelated.read_bytes() == b"keep"
