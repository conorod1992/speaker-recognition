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


class EnrollmentUpdateFailed(RuntimeError):
    """Raised when changed enrollment samples fail to replace a profile."""

    def __init__(
        self,
        changed_users: set[str],
        previous_samples: list[dict[str, Any]],
    ) -> None:
        super().__init__("Speaker enrollment update failed")
        self.changed_users = changed_users
        self.previous_samples = previous_samples


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
) -> bool:
    """Load backend status and process only a newly completed enrollment.

    Return whether there is no pending enrollment left to retry.
    """
    await recognition.async_refresh_status()
    if pending_user is None:
        return True
    return await recognition.async_train({pending_user})


async def async_apply_enrollment_update(
    recognition: RecognitionLifecycle,
    voice_samples: list[dict[str, Any]],
) -> set[str]:
    """Train only users whose configured samples actually changed."""
    previous_samples = recognition.voice_samples
    previous = _samples_by_user(previous_samples)
    current = _samples_by_user(voice_samples)
    changed_users = {
        user_id
        for user_id, sample in current.items()
        if previous.get(user_id) != sample
    }
    recognition.update_voice_samples(voice_samples)
    if changed_users and not await recognition.async_train(changed_users):
        recognition.update_voice_samples(previous_samples)
        raise EnrollmentUpdateFailed(changed_users, previous_samples)
    return changed_users
