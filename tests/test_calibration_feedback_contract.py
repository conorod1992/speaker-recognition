"""Contract coverage for audio-free calibration telemetry and feedback."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"


def test_history_is_bounded_persistent_and_privacy_preserving() -> None:
    """Decision history stays small and never stores spoken content."""
    source = (ROOT / "telemetry.py").read_text(encoding="utf-8")
    record_schema = source.split("class DecisionRecord:", 1)[1].split(
        "class DecisionHistory:", 1
    )[0]

    assert "_MAX_DECISIONS = 200" in source
    assert "async_delay_save" in source
    assert 'f"{DOMAIN}.decision_history"' in source
    assert "recognition_seconds" in record_schema
    assert "added_latency_seconds" in record_schema
    assert "satellite_id" in record_schema
    assert "utterance_sequence" in record_schema
    assert "stt_entity_id" in record_schema
    assert "transcript:" not in record_schema
    assert "text:" not in record_schema
    assert "audio_data:" not in record_schema
    assert "pcm_audio:" not in record_schema


def test_stt_turns_are_recorded_without_requiring_conversation_proxy() -> None:
    """Every normal SR STT result feeds calibration history directly."""
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    telemetry_source = (ROOT / "telemetry.py").read_text(encoding="utf-8")

    assert '"speaker_recognition_detected"' in init_source
    assert "history.record_event" in init_source
    assert "def record_event" in telemetry_source
    assert "_find_turn" in telemetry_source
    assert "utterance_sequence" in telemetry_source
    assert "stt_entity_id" in telemetry_source


def test_enrollment_turns_are_excluded_from_stt_calibration_history() -> None:
    """Prompted enrollment phrases never appear as normal calibration decisions."""
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    enrollment_source = (ROOT / "enrollment.py").read_text(encoding="utf-8")

    assert '"calibration_excluded_utterances"' in enrollment_source
    assert '"calibration_excluded_utterances"' in init_source
    assert "excluded.discard(sequence)" in init_source
    assert "history.record_event" in init_source


def test_conversation_proxy_enriches_instead_of_duplicating_stt_record() -> None:
    """Conversation context evaluation updates the already-recorded STT turn."""
    telemetry_source = (ROOT / "telemetry.py").read_text(encoding="utf-8")
    conversation_source = (ROOT / "conversation.py").read_text(encoding="utf-8")

    assert "existing = self._find_turn" in telemetry_source
    assert '"identity_eligible": identity_eligible' in telemetry_source
    assert '"threshold": threshold' in telemetry_source
    assert "history.record(" in conversation_source


def test_feedback_requires_ground_truth_for_errors() -> None:
    """Wrong/missed decisions require the actual HA user to be supplied."""
    source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert '"wrong_speaker"' in source
    assert '"missed_speaker"' in source
    assert '"actual_user_required"' in source
    assert '"unknown_user"' in source
    assert 'f"{DOMAIN}/decision_history"' in source
    assert 'f"{DOMAIN}/decision_feedback"' in source


def test_frontend_exposes_recent_decisions_and_feedback() -> None:
    """The panel makes collected calibration evidence understandable and editable."""
    source = (ROOT / "www" / "speaker-recognition-panel.js").read_text(
        encoding="utf-8"
    )
    calibration = (
        ROOT / "www" / "speaker-recognition-calibration-panel.js"
    ).read_text(encoding="utf-8")

    assert "Recognition calibration" in source
    assert "Recent normal Assist decisions are stored without audio or transcripts" in source
    assert "Correct" in source
    assert "Wrong speaker" in source
    assert "Should have recognised me" in source
    assert "Actual speaker for corrections" in source
    assert "speaker_recognition/decision_feedback" in source
    assert "recent recognition decision" in calibration
    assert "STT proxy even when your pipeline sends the conversation directly" in calibration


def test_changed_ha_modules_compile() -> None:
    """Catch Python syntax errors in HA files excluded from the backend lint target."""
    for name in (
        "__init__.py",
        "conversation.py",
        "enrollment.py",
        "telemetry.py",
        "websocket.py",
    ):
        source = (ROOT / name).read_text(encoding="utf-8")
        compile(source, str(ROOT / name), "exec")
