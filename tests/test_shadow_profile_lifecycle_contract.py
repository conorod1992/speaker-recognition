"""Lightweight tests for shadow profile lifecycle behavior."""

from __future__ import annotations

from pathlib import Path


def test_shadow_profiles_are_isolated_from_authoritative_profiles() -> None:
    shadow_source = Path("speaker_recognition/shadow.py").read_text(encoding="utf-8")
    assert 'Path(config.embeddings_directory) / "shadow" / engine.info.engine_id' in shadow_source
    assert "SpeakerRecognizer(config=shadow_config, engine=engine)" in shadow_source


def test_health_does_not_take_shadow_lock() -> None:
    api = Path("speaker_recognition/api.py").read_text(encoding="utf-8")
    health_start = api.index("def health_check")
    next_endpoint = api.index("@app.post", health_start)
    health_source = api[health_start:next_endpoint]
    assert "_RECOGNIZER_LOCK" in health_source
    assert "_SHADOW_LOCK" not in health_source
