"""Enrollment lifecycle decisions independent of Home Assistant plumbing."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class RecognitionLifecycle(Protocol):
    """Operations needed to initialize and update recognition state."""

    voice_samples: list[dict[str, Any]]

    @property
    def enrolled_users(self) -> set[str]:
        """Return persisted backend profile IDs."""

    @property
    def configured_users(self) -> set[str]:
        """Return configured enrollment user IDs."""

    async def async_refresh_status(self) -> bool:
        """Refresh availability from persisted backend profiles."""

    async def async_train(self, user_ids: set[str] | None = None) -> bool:
        """Train configured samples for selected users."""

    async def async_sync_profiles(self) -> bool:
        """Remove persisted profiles not present in current configuration."""

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


class ProfileReconciliationFailed(RuntimeError):
    """Raised when configured profiles cannot be restored at startup."""


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
    """Reconcile configured users with persisted backend profiles."""
    await recognition.async_refresh_status()
    configured = recognition.configured_users
    missing = configured - recognition.enrolled_users
    if pending_user is not None and pending_user in configured:
        missing.add(pending_user)

    if missing and not await recognition.async_train(missing):
        raise ProfileReconciliationFailed(
            "Configured speaker profiles could not be restored"
        )

    # Stale profiles are not allowed to remain identity candidates. If cleanup
    # itself fails, runtime identity validation still rejects unconfigured IDs,
    # and the next setup/update retries synchronization.
    await recognition.async_sync_profiles()
    return pending_user is None or pending_user in recognition.enrolled_users


async def async_apply_enrollment_update(
    recognition: RecognitionLifecycle,
    voice_samples: list[dict[str, Any]],
) -> set[str]:
    """Train changed users, then synchronize removed profiles."""
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

    # Do not roll back a successfully trained configuration merely because stale
    # profile cleanup failed. The new profile is already authoritative, and
    # unconfigured backend identities are rejected by runtime validation.
    await recognition.async_sync_profiles()
    return changed_users | (set(previous) - set(current))
