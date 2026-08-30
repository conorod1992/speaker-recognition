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
    """Record lifecycle calls while modeling backend profiles."""

    def __init__(
        self,
        voice_samples,
        *,
        enrolled_users: set[str] | None = None,
        training_succeeds: bool = True,
        sync_succeeds: bool = True,
    ) -> None:
        self.voice_samples = voice_samples
        self._enrolled_users = set(enrolled_users or set())
        self.training_succeeds = training_succeeds
        self.sync_succeeds = sync_succeeds
        self.status_refreshes = 0
        self.training_calls: list[set[str] | None] = []
        self.sync_calls = 0

    @property
    def configured_users(self) -> set[str]:
        return {
            sample["user"]
            for sample in self.voice_samples
            if isinstance(sample.get("user"), str)
        }

    @property
    def enrolled_users(self) -> set[str]:
        return set(self._enrolled_users)

    async def async_refresh_status(self) -> bool:
        self.status_refreshes += 1
        return bool(self.configured_users & self._enrolled_users)

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        self.training_calls.append(user_ids)
        if self.training_succeeds:
            self._enrolled_users.update(user_ids or self.configured_users)
        return self.training_succeeds

    async def async_sync_profiles(self) -> bool:
        self.sync_calls += 1
        if self.sync_succeeds:
            self._enrolled_users.intersection_update(self.configured_users)
        return self.sync_succeeds

    def update_voice_samples(self, voice_samples) -> None:
        self.voice_samples = voice_samples


def _sample(user: str, media: str) -> dict:
    return {"user": user, "samples": [{"media_content_id": media}]}


@pytest.mark.asyncio
async def test_restart_uses_complete_persisted_profiles_without_retraining() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [_sample("alice", "media-source://old")], enrolled_users={"alice"}
    )
    completed = await lifecycle.async_initialize_recognition(recognition)
    assert completed
    assert recognition.status_refreshes == 1
    assert recognition.training_calls == []
    assert recognition.sync_calls == 1


@pytest.mark.asyncio
async def test_restart_restores_only_missing_configured_profiles() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [
            _sample("alice", "media-source://alice"),
            _sample("bob", "media-source://bob"),
        ],
        enrolled_users={"alice"},
    )
    completed = await lifecycle.async_initialize_recognition(recognition)
    assert completed
    assert recognition.training_calls == [{"bob"}]
    assert recognition.enrolled_users == {"alice", "bob"}


@pytest.mark.asyncio
async def test_restart_removes_stale_backend_profiles() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [_sample("alice", "media-source://alice")],
        enrolled_users={"alice", "deleted-user"},
    )
    assert await lifecycle.async_initialize_recognition(recognition)
    assert recognition.training_calls == []
    assert recognition.sync_calls == 1
    assert recognition.enrolled_users == {"alice"}


@pytest.mark.asyncio
async def test_missing_profile_failure_blocks_inconsistent_startup() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [_sample("alice", "media-source://alice")],
        enrolled_users=set(),
        training_succeeds=False,
    )
    with pytest.raises(lifecycle.ProfileReconciliationFailed):
        await lifecycle.async_initialize_recognition(recognition)
    assert recognition.training_calls == [{"alice"}]
    assert recognition.sync_calls == 0


@pytest.mark.asyncio
async def test_new_guided_enrollment_trains_once_during_initial_setup() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [_sample("alice", "media-source://new-alice")], enrolled_users=set()
    )
    completed = await lifecycle.async_initialize_recognition(recognition, "alice")
    assert completed
    assert recognition.training_calls == [{"alice"}]
    assert recognition.sync_calls == 1


@pytest.mark.asyncio
async def test_failed_pending_enrollment_remains_incomplete() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [_sample("alice", "media-source://bad-alice")],
        enrolled_users=set(),
        training_succeeds=False,
    )
    with pytest.raises(lifecycle.ProfileReconciliationFailed):
        await lifecycle.async_initialize_recognition(recognition, "alice")


@pytest.mark.asyncio
async def test_changed_enrollment_trains_only_updated_user() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [
            _sample("alice", "media-source://old-alice"),
            _sample("bob", "media-source://old-bob"),
        ],
        enrolled_users={"alice", "bob"},
    )
    updated = [
        _sample("alice", "media-source://new-alice"),
        _sample("bob", "media-source://old-bob"),
    ]
    changed = await lifecycle.async_apply_enrollment_update(recognition, updated)
    assert changed == {"alice"}
    assert recognition.training_calls == [{"alice"}]
    assert recognition.sync_calls == 1
    assert recognition.voice_samples == updated


@pytest.mark.asyncio
async def test_removed_user_is_synchronized_without_retraining() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    recognition = _FakeRecognition(
        [
            _sample("alice", "media-source://alice"),
            _sample("bob", "media-source://bob"),
        ],
        enrolled_users={"alice", "bob"},
    )
    changed = await lifecycle.async_apply_enrollment_update(
        recognition, [_sample("alice", "media-source://alice")]
    )
    assert changed == {"bob"}
    assert recognition.training_calls == []
    assert recognition.sync_calls == 1
    assert recognition.enrolled_users == {"alice"}


@pytest.mark.asyncio
async def test_failed_retraining_restores_previous_runtime_configuration() -> None:
    lifecycle = _load_integration_module("lifecycle.py")
    original = [_sample("bob", "media-source://old-bob")]
    recognition = _FakeRecognition(
        original, enrolled_users={"bob"}, training_succeeds=False
    )
    updated = [_sample("bob", "media-source://new-bob")]
    with pytest.raises(lifecycle.EnrollmentUpdateFailed) as raised:
        await lifecycle.async_apply_enrollment_update(recognition, updated)
    assert raised.value.changed_users == {"bob"}
    assert raised.value.previous_samples == original
    assert recognition.training_calls == [{"bob"}]
    assert recognition.sync_calls == 0
    assert recognition.voice_samples == original


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_roll_back_successful_retraining() -> None:
    """A stale-delete outage cannot put HA config behind an already-updated profile."""
    lifecycle = _load_integration_module("lifecycle.py")
    original = [
        _sample("alice", "media-source://old-alice"),
        _sample("bob", "media-source://bob"),
    ]
    updated = [_sample("alice", "media-source://new-alice")]
    recognition = _FakeRecognition(
        original,
        enrolled_users={"alice", "bob"},
        sync_succeeds=False,
    )
    changed = await lifecycle.async_apply_enrollment_update(recognition, updated)
    assert changed == {"alice", "bob"}
    assert recognition.voice_samples == updated
    assert recognition.training_calls == [{"alice"}]
    assert recognition.sync_calls == 1


def test_effective_backend_url_prefers_current_options() -> None:
    constants = _load_integration_module("const.py")
    assert (
        constants.effective_backend_url(
            {constants.CONF_BACKEND_URL: "http://original:8099"},
            {constants.CONF_BACKEND_URL: "http://current:8099"},
        )
        == "http://current:8099"
    )
