"""Contract coverage for the ten-item playable calibration review queue."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"


def test_review_audio_is_separate_bounded_and_does_not_pollute_decision_schema() -> None:
    source = (ROOT / "telemetry.py").read_text(encoding="utf-8")
    record_schema = source.split("class DecisionRecord:", 1)[1].split(
        "class DecisionHistory:", 1
    )[0]

    assert '_AUDIO_STORAGE_KEY = f"{DOMAIN}.decision_review_audio"' in source
    assert "_MAX_REVIEW_AUDIO = 10" in source
    assert "_MAX_REVIEW_AUDIO_SECONDS = 30" in source
    assert "self._review_audio = self._review_audio[-_MAX_REVIEW_AUDIO:]" in source
    assert "def review_recent" in source
    assert 'item["has_audio"]' in source
    assert "pcm_base64" not in record_schema
    assert "audio_data" not in record_schema
    assert "transcript" not in record_schema


def test_review_audio_is_captured_from_the_existing_correlated_pcm_cache() -> None:
    source = (ROOT / "telemetry.py").read_text(encoding="utf-8")

    assert 'get("utterance_audio")' in source
    assert "def _capture_review_audio" in source
    assert "self._capture_review_audio(item)" in source
    assert "self._capture_review_audio(existing)" in source
    assert "base64.b64encode(bounded_pcm)" in source


def test_review_websocket_exposes_only_ten_decisions_and_lazy_audio() -> None:
    source = (ROOT / "review_audio_websocket.py").read_text(encoding="utf-8")

    assert "_REVIEW_LIMIT = 10" in source
    assert 'f"{DOMAIN}/review_decisions"' in source
    assert "history.review_recent(_REVIEW_LIMIT)" in source
    assert 'f"{DOMAIN}/decision_audio"' in source
    assert "history.review_audio_for_decision" in source
    assert '"audio_expired"' in source
    assert 'f"{DOMAIN}/review_feedback"' in source
    assert "websocket_review_feedback" in source


def test_calibration_ui_defaults_to_listen_decide_and_hides_diagnostics() -> None:
    source = (
        ROOT / "www" / "speaker-recognition-calibration-panel.js"
    ).read_text(encoding="utf-8")

    assert 'type: "speaker_recognition/review_decisions"' in source
    assert 'type: "speaker_recognition/decision_audio"' in source
    assert "▶ Play clip" in source
    assert "<summary>Diagnostics</summary>" in source
    assert "Correctly unknown" in source
    assert "That was me" in source
    assert "Not me" in source
    assert "newest ten Assist decisions" in source
    assert "oldest clip is discarded automatically" in source


def test_review_feedback_supports_not_enrolled_ground_truth() -> None:
    websocket = (ROOT / "review_audio_websocket.py").read_text(encoding="utf-8")
    frontend = (
        ROOT / "www" / "speaker-recognition-calibration-panel.js"
    ).read_text(encoding="utf-8")

    assert 'vol.Optional("actual_user_id"): vol.Any(str, None)' in websocket
    assert 'feedback == "missed_speaker"' in websocket
    assert 'feedback == "correct"' in websocket
    assert "__unknown__" in frontend
    assert "Someone not enrolled" in frontend


def test_review_audio_python_modules_compile() -> None:
    for name in ("telemetry.py", "review_audio_websocket.py"):
        path = ROOT / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
