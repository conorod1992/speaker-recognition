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


def test_satellite_enrollment_is_bound_to_exact_satellite_turn() -> None:
    """Satellite capture uses turn correlation and a precise capture session."""
    stt_source = (ROOT / "stt.py").read_text(encoding="utf-8")
    conversation_source = (ROOT / "conversation.py").read_text(encoding="utf-8")
    enrollment_source = (ROOT / "enrollment.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")
    panel = (ROOT / "www" / "speaker-recognition-panel.js").read_text(encoding="utf-8")

    assert 'domain_data.setdefault("utterance_audio", {})' in stt_source
    assert "speaker_data.utterance_sequence" in conversation_source
    assert "async_capture_satellite_sample" in conversation_source
    assert "sessions.get(satellite_id)" in enrollment_source
    assert "session.session_id" in enrollment_source
    assert "AssistSatelliteEntityFeature.START_CONVERSATION" in websocket_source
    assert '"session_id": session_id' in websocket_source
    assert "completed_satellite_captures" in panel
    assert "completed.includes(sessionId)" in panel


def test_interactive_enrollment_reuses_transactional_profile_update() -> None:
    """Staged samples use the config-entry listener without duplicate startup training."""
    init_source = (ROOT / "__init__.py").read_text(encoding="utf-8")
    websocket_source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert "CONF_PENDING_ENROLLMENT" not in websocket_source
    assert "async_update_entry(entry, options=options)" in websocket_source
    assert "async_apply_enrollment_update" in init_source
    assert 'staged.pop(user_id, None)' in init_source
