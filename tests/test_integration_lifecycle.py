"""Focused tests for Home Assistant recognition lifecycle decisions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_integration_module(filename: str):
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / filename
    )
    spec = importlib.util.spec_from_file_location(
        f"speaker_recognition_integration_{module_path.stem}", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRecognition:
    """Record lifecycle calls while modeling existing backend usability."""

    def __init__(self, voice_samples, *, training_succeeds: bool = True) -> None:
        self.voice_samples = voice_samples
        self.training_succeeds = training_succeeds
        self.usable_profiles = True
        self.status_refreshes = 0
        self.training_calls: list[set[str] | None] = []

    async def async_refresh_status(self) -> bool:
        self.status_refreshes += 1
        return self.usable_profiles

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        self.training_calls.append(user_ids)
        if self.training_succeeds:
            self.usable_profiles = True
        return self.training_succeeds

    def update_voice_samples(self, voice_samples) -> None:
        self.voice_samples = voice_samples


@pytest.mark.asyncio
async def test_restart_uses_persisted_status_without_retraining() -> None:
    """Ordinary setup refreshes backend status and never reads enrollment media."""
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [{"user": "alice", "samples": [{"media_content_id": "media-source://old"}]}]
    )

    await lifecycle.async_initialize_recognition(recognition)

    assert recognition.status_refreshes == 1
    assert recognition.training_calls == []
    assert recognition.usable_profiles


@pytest.mark.asyncio
async def test_new_guided_enrollment_trains_once_during_initial_setup() -> None:
    """The explicit pending marker processes a newly completed initial enrollment."""
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [{"user": "alice", "samples": [{"media_content_id": "new-alice"}]}]
    )

    await lifecycle.async_initialize_recognition(recognition, "alice")

    assert recognition.status_refreshes == 1
    assert recognition.training_calls == [{"alice"}]


@pytest.mark.asyncio
async def test_changed_enrollment_trains_only_updated_user() -> None:
    """Completing guided enrollment still invokes training for that user."""
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [
            {"user": "alice", "samples": [{"media_content_id": "old-alice"}]},
            {"user": "bob", "samples": [{"media_content_id": "old-bob"}]},
        ]
    )
    updated = [
        {"user": "alice", "samples": [{"media_content_id": "new-alice"}]},
        {"user": "bob", "samples": [{"media_content_id": "old-bob"}]},
    ]

    changed = await lifecycle.async_apply_enrollment_update(recognition, updated)

    assert changed == {"alice"}
    assert recognition.training_calls == [{"alice"}]


@pytest.mark.asyncio
async def test_failed_retraining_preserves_existing_backend_usability() -> None:
    """A failed changed enrollment does not invalidate previously loaded profiles."""
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [{"user": "bob", "samples": [{"media_content_id": "old-bob"}]}],
        training_succeeds=False,
    )

    await lifecycle.async_apply_enrollment_update(
        recognition,
        [
            {"user": "bob", "samples": [{"media_content_id": "old-bob"}]},
            {"user": "alice", "samples": [{"media_content_id": "bad-alice"}]},
        ],
    )

    assert recognition.training_calls == [{"alice"}]
    assert recognition.usable_profiles


def test_effective_backend_url_prefers_current_options() -> None:
    """Reopening options displays the same URL used by runtime setup."""
    constants = _load_integration_module("const.py")

    assert (
        constants.effective_backend_url(
            {constants.CONF_BACKEND_URL: "http://original:8099"},
            {constants.CONF_BACKEND_URL: "http://current:8099"},
        )
        == "http://current:8099"
    )
