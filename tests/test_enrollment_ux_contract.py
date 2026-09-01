"""Contract tests for interactive enrollment and diagnostics."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"


def test_frontend_panel_is_registered_with_websocket_commands() -> None:
    """The integration exposes its panel and dedicated enrollment API."""
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    frontend_source = (ROOT / "frontend.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "async_register_frontend(hass)" in init_source
    assert "async_register_websocket_commands(hass)" in init_source
    assert "panel_custom.async_register_panel" in frontend_source
    assert 'f"{DOMAIN}/stage_sample"' in websocket_source
    assert 'f"{DOMAIN}/test_sample"' in websocket_source


def test_browser_panel_keeps_upload_fallback_and_secure_context_message() -> None:
    """Browser recording is feature-detected instead of assumed available."""
    panel = (ROOT / "www" / "speaker-recognition-panel.js").read_text(encoding="utf-8")

    assert "navigator.mediaDevices.getUserMedia" in panel
    assert "window.isSecureContext" in panel
    assert "existing WAV upload flow" in panel
    assert "Test profile" in panel


def test_selected_user_enrollment_status_is_explicit() -> None:
    """The panel explains whether the selected user is new, enrolled or retraining."""
    panel = (
        ROOT / "www" / "speaker-recognition-calibration-panel.js"
    ).read_text(encoding="utf-8")

    assert "Selected user status" in panel
    assert "Current profile" in panel
    assert "Enrollment mode" in panel
    assert "Retraining existing profile" in panel
    assert "New enrollment" in panel
    assert "Ready to train" in panel
    assert "existing trained profile stays in use" in panel


def test_satellite_enrollment_is_claimed_during_selected_satellite_stt() -> None:
    """Enrollment completion no longer depends on the Conversation proxy."""
    stt_source = (ROOT / "stt.py").read_text(encoding="utf-8")
    enrollment_source = (ROOT / "enrollment.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")
    panel = (ROOT / "www" / "speaker-recognition-panel.js").read_text(encoding="utf-8")

    assert "claim_satellite_enrollment_turn" in stt_source
    assert "async_capture_claimed_satellite_sample" in stt_source
    assert 'state.state == "listening"' in enrollment_source
    assert "claimed_utterance_sequence" in enrollment_source
    assert "_completed_satellite_captures(hass)[session.session_id]" in enrollment_source
    assert "AssistSatelliteEntityFeature.START_CONVERSATION" in websocket_source
    assert '"session_id": session_id' in websocket_source
    assert "completed_satellite_captures" in panel
    assert "completed.includes(sessionId)" in panel


def test_enrollment_has_priority_over_a_live_test_for_the_same_stt_turn() -> None:
    """A prompted enrollment utterance cannot accidentally satisfy a live test."""
    stt_source = (ROOT / "stt.py").read_text(encoding="utf-8")

    assert "if not enrollment_turn_claimed:" in stt_source
    assert "live_test_claimed = claim_live_test_turn" in stt_source


def test_interactive_enrollment_reuses_transactional_profile_update() -> None:
    """Staged samples use the config-entry listener without duplicate startup training."""
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "CONF_PENDING_ENROLLMENT" not in websocket_source
    assert "async_update_entry(entry, options=options)" in websocket_source
    assert "async_apply_enrollment_update" in init_source
    assert 'staged.pop(user_id, None)' in init_source


def test_retake_cleans_the_superseded_staged_media_file() -> None:
    """Replacing one staged phrase does not leak the previous managed WAV."""
    enrollment_source = (ROOT / "enrollment.py").read_text(encoding="utf-8")

    assert "previous = staged.get(sample_index)" in enrollment_source
    assert "previous_path = _managed_media_path(hass, previous_media_id)" in enrollment_source
    assert "previous_path != absolute_path" in enrollment_source
    assert "_delete_paths, [previous_path]" in enrollment_source


def test_enrollment_sample_bounds_follow_the_phrase_catalogue() -> None:
    """Adding enrollment phrases does not require updating hard-coded API bounds."""
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "_ENROLLMENT_SAMPLE_INDEX = vol.All(" in websocket_source
    assert "max=len(ENROLLMENT_PHRASES) - 1" in websocket_source
    assert "vol.Range(min=0, max=5)" not in websocket_source
    assert websocket_source.count(
        'vol.Required("sample_index"): _ENROLLMENT_SAMPLE_INDEX'
    ) == 2


def test_commit_metadata_preserves_exact_staged_phrase_indexes() -> None:
    """Non-contiguous staged samples retain the phrase they actually recorded."""
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "ordered = [(index, staged[index]) for index in sorted(staged)]" in websocket_source
    assert '"sample_index": sample_index' in websocket_source
    assert '"phrase": ENROLLMENT_PHRASES[sample_index]' in websocket_source
    assert "ENROLLMENT_PHRASES[: len(samples)]" not in websocket_source


def test_panel_advances_to_the_next_incomplete_phrase() -> None:
    """Sparse staged indexes do not make the panel skip or revisit the wrong phrase."""
    panel = (ROOT / "www" / "speaker-recognition-panel.js").read_text(encoding="utf-8")

    assert "_nextIncompleteSample(afterIndex = -1)" in panel
    assert "new Set(this._stagedIndexes())" in panel
    assert "this._sampleIndex = this._nextIncompleteSample(savedIndex)" in panel
    assert "this._sampleIndex = this._nextIncompleteSample(index)" in panel
    assert "Math.min(staged.length" not in panel
