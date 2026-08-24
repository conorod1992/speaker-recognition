"""Enrollment lifecycle decisions independent of Home Assistant plumbing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class RecognitionLifecycle(Protocol):
    """Operations needed to initialize and update recognition state."""

    voice_samples: list[dict[str, Any]]

    async def async_refresh_status(self) -> bool:
        """Refresh availability from persisted backend profiles."""

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        """Train configured samples for selected users."""

    def update_voice_samples(self, voice_samples: list[dict[str, Any]]) -> None:
        """Replace configured media references."""


def _samples_by_user(
    voice_samples: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index valid enrollment configuration by user ID."""
    return {
        user_id: sample
        for sample in voice_samples
        if isinstance((user_id := sample.get("user")), str)
    }


async def async_initialize_recognition(
    recognition: RecognitionLifecycle, pending_user: str | None = None
) -> None:
    """Load backend status and process only a newly completed enrollment."""
    await recognition.async_refresh_status()
    if pending_user is not None:
        await recognition.async_train({pending_user})


async def async_apply_enrollment_update(
    recognition: RecognitionLifecycle,
    voice_samples: list[dict[str, Any]],
) -> set[str]:
    """Train only users whose configured samples actually changed."""
    previous = _samples_by_user(recognition.voice_samples)
    current = _samples_by_user(voice_samples)
    changed_users = {
        user_id
        for user_id, sample in current.items()
        if previous.get(user_id) != sample
    }
    recognition.update_voice_samples(voice_samples)
    if changed_users:
        await recognition.async_train(changed_users)
    return changed_users
