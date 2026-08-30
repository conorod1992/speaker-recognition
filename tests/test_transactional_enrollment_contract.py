"""Regression contracts for transactional sidebar enrollment."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
HA = ROOT / "custom_components" / "speaker_recognition"


def test_staged_recordings_use_unique_media_paths() -> None:
    source = (HA / "enrollment.py").read_text(encoding="utf-8")
    assert "generation = secrets.token_urlsafe" in source
    assert "sample_{sample_index + 1}_{generation}.wav" in source
    assert 'sample_{sample_index + 1}.wav"' not in source


def test_commit_waits_for_backend_training_result() -> None:
    source = (HA / "websocket.py").read_text(encoding="utf-8")
    assert "@websocket_api.async_response" in source
    assert '"enrollment_commit_waiters"' in source
    assert "await asyncio.wait_for(asyncio.shield(completion)" in source
    assert '"training_failed"' in source
    assert '"training_timeout"' in source


def test_update_listener_reports_training_success_or_failure() -> None:
    source = (HA / "__init__.py").read_text(encoding="utf-8")
    assert "def _resolve_enrollment_commits(" in source
    assert "_resolve_enrollment_commits(hass, error.changed_users, False)" in source
    assert "_resolve_enrollment_commits(hass, changed_users, True)" in source
