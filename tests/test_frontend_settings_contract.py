"""Contract coverage for the custom frontend settings surface."""

from pathlib import Path
import py_compile

ROOT = Path(__file__).parents[1]
HA = ROOT / "custom_components" / "speaker_recognition"


def test_settings_websocket_exposes_all_existing_user_configuration() -> None:
    source = (HA / "settings_websocket.py").read_text(encoding="utf-8")
    assert 'f"{DOMAIN}/settings"' in source
    assert 'f"{DOMAIN}/update_settings"' in source
    assert "CONF_BACKEND_URL" in source
    assert "CONF_STT_ENTITY" in source
    assert "CONF_USE_BASIC_DSP" in source
    assert "CONF_CONVERSATION_ENTITY" in source
    assert "CONF_MIN_CONFIDENCE" in source


def test_frontend_is_registered_to_settings_panel() -> None:
    frontend = (HA / "frontend.py").read_text(encoding="utf-8")
    panel = (HA / "www" / "speaker-recognition-settings-panel.js").read_text(
        encoding="utf-8"
    )
    assert 'PANEL_ELEMENT = "speaker-recognition-settings-panel"' in frontend
    assert "speaker-recognition-settings-panel.js" in frontend
    assert "Use basic DSP for speech-to-text" in panel
    assert "Backend URL" in panel
    assert "Minimum identity confidence" in panel
    assert "speaker_recognition/update_settings" in panel


def test_settings_backend_module_compiles() -> None:
    py_compile.compile(str(HA / "settings_websocket.py"), doraise=True)
