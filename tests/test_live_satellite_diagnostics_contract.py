"""Contract tests for live satellite recognition diagnostics."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"


def test_changed_home_assistant_modules_compile() -> None:
    """Catch syntax errors without importing Home Assistant in the test environment."""
    for filename in (
        "conversation.py",
        "correlation.py",
        "diagnostics.py",
        "stt.py",
        "websocket.py",
        "whisper.py",
    ):
        source = (ROOT / filename).read_text(encoding="utf-8")
        compile(source, str(ROOT / filename), "exec")


def test_live_test_is_bound_to_selected_satellite_without_intercepting_assist() -> None:
    """A live test observes one exact normal Assist turn from the chosen satellite."""
    diagnostics_source = (ROOT / "diagnostics.py").read_text(encoding="utf-8")
    conversation_source = (ROOT / "conversation.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "satellite_id != session.satellite_id" in diagnostics_source
    assert "record_live_test_result" in conversation_source
    assert 'f"{DOMAIN}/start_live_test"' in websocket_source
    assert "start_conversation" not in websocket_source.split(
        "def websocket_start_live_test", 1
    )[1].split("def websocket_commit_enrollment", 1)[0]


def test_turn_correlation_exposes_real_latency_breakdown() -> None:
    """Diagnostics distinguish recognition duration from actual Assist delay."""
    correlation_source = (ROOT / "correlation.py").read_text(encoding="utf-8")
    stt_source = (ROOT / "stt.py").read_text(encoding="utf-8")
    diagnostics_source = (ROOT / "diagnostics.py").read_text(encoding="utf-8")

    for field in (
        "stt_seconds",
        "recognition_seconds",
        "preparation_seconds",
        "added_latency_seconds",
        "audio_seconds",
    ):
        assert field in correlation_source
        assert field in diagnostics_source

    assert "recognition_completed_at - stt_completed_at" in stt_source
    assert "max(0.0" in stt_source


def test_live_test_exposes_whispering_independently_of_speaker_identity() -> None:
    """A rejected/unknown speaker can still produce a generic whisper result."""
    correlation_source = (ROOT / "correlation.py").read_text(encoding="utf-8")
    diagnostics_source = (ROOT / "diagnostics.py").read_text(encoding="utf-8")
    stt_source = (ROOT / "stt.py").read_text(encoding="utf-8")
    panel = (ROOT / "www" / "speaker-recognition-calibration-panel.js").read_text(
        encoding="utf-8"
    )

    for field in ("whispering", "whisper_score", "whisper_available"):
        assert field in correlation_source
        assert field in diagnostics_source
        assert field in stt_source

    assert "detect_whisper" in stt_source
    assert "asyncio.gather" in stt_source
    assert '"whispering": correlated.whispering' in stt_source
    assert '"whispering": recognition.whispering' in diagnostics_source
    assert "Whispering Detected" in panel
    assert 'result.whispering ? "Yes" : "No"' in panel


def test_frontend_explains_live_test_and_cold_first_recognition() -> None:
    """The panel explains realistic testing and first-use warm-up."""
    panel = (ROOT / "www" / "speaker-recognition-panel.js").read_text(
        encoding="utf-8"
    )

    assert "Live satellite test" in panel
    assert "Start live test" in panel
    assert "Added Assist latency" in panel
    assert "first recognition after the backend starts can take longer" in panel
    assert "normal Assist request still runs" in panel
