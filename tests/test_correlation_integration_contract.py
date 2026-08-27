"""Integration-contract checks for correlated speaker decisions."""

from pathlib import Path


def test_correlated_decision_carries_stt_turn_identity() -> None:
    """Correlation state retains the STT proxy and utterance sequence."""
    root = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
    source = (root / "correlation.py").read_text(encoding="utf-8")

    assert "stt_entity_id: str | None" in source
    assert "utterance_sequence: int" in source


def test_conversation_logs_pipeline_device_context() -> None:
    """The consumer joins task-local recognition with device/satellite metadata."""
    root = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
    source = (root / "conversation.py").read_text(encoding="utf-8")

    assert "user_input.device_id" in source
    assert "user_input.satellite_id" in source
    assert "speaker_data.utterance_sequence" in source
    assert "speaker_data.stt_entity_id" in source
