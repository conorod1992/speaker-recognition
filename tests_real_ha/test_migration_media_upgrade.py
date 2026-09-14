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
    CONF_ENTRY_TYPE,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)


@pytest.mark.asyncio
async def test_legacy_main_entry_migrates_to_current_version_without_losing_samples(
    hass: HomeAssistant,
) -> None:
    """A v1 main entry upgrades in place while preserving enrollment references."""
    samples = {"alice": ["/config/speaker_recognition/alice-1.wav"]}
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: "secret",
        },
        options={CONF_VOICE_SAMPLES: samples},
        unique_id=ENTRY_TYPE_MAIN,
        version=1,
    )
    entry.add_to_hass(hass)

    assert await async_migrate_entry(hass, entry) is True
    assert entry.version == 2
    assert entry.minor_version == 0
    assert entry.data[CONF_ENTRY_TYPE] == ENTRY_TYPE_MAIN
    assert entry.options[CONF_VOICE_SAMPLES] == samples


@pytest.mark.asyncio
async def test_current_entry_migration_is_idempotent(hass: HomeAssistant) -> None:
    """Re-running migration on a current entry must not rewrite user data."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: "http://127.0.0.1:8099",
            CONF_BACKEND_TOKEN: "",
        },
        options={CONF_VOICE_SAMPLES: {"alice": ["sample.wav"]}},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )
    entry.add_to_hass(hass)
    before_data = dict(entry.data)
    before_options = dict(entry.options)

    assert await async_migrate_entry(hass, entry) is True
    assert dict(entry.data) == before_data
    assert dict(entry.options) == before_options


def test_managed_enrollment_paths_cannot_escape_storage_root(tmp_path: Path) -> None:
    """Resolved managed sample targets stay below the integration-owned directory."""
    root = (tmp_path / "speaker_recognition").resolve()
    root.mkdir()
    candidate = (root / "user" / "sample.wav").resolve()
    candidate.parent.mkdir()

    assert candidate.is_relative_to(root)
    assert not (root / ".." / "outside.wav").resolve().is_relative_to(root)
