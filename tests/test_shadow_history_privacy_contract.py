"""Contract that shadow comparison remains audio-free in persistent history."""

from pathlib import Path


def test_shadow_history_persists_scores_not_audio() -> None:
    source = Path("custom_components/speaker_recognition/telemetry.py").read_text(
        encoding="utf-8"
    )
    assert "shadow_all_scores" in source
    assert "shadow_processing_seconds" in source
    decision_class = source[source.index("class DecisionRecord"):source.index("class DecisionHistory")]
    assert "audio_data" not in decision_class
    assert "transcript" not in decision_class
