"""Contract coverage for threshold guidance and manual application."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"


def test_websocket_exposes_analysis_and_manual_apply() -> None:
    source = (ROOT / "websocket.py").read_text(encoding="utf-8")

    assert 'f"{DOMAIN}/calibration_analysis"' in source
    assert 'f"{DOMAIN}/apply_recommended_threshold"' in source
    assert "analyze_thresholds(history.labelled(), current_threshold)" in source
    assert "options[CONF_MIN_CONFIDENCE] = float(recommendation)" in source
    assert '"insufficient_evidence"' in source
    assert "vol.Required(\"entry_id\")" in source


def test_frontend_requires_explicit_apply_and_explains_limitations() -> None:
    frontend = (ROOT / "frontend.py").read_text(encoding="utf-8")
    panel = (ROOT / "www" / "speaker-recognition-calibration-panel.js").read_text(
        encoding="utf-8"
    )

    assert 'PANEL_ELEMENT = "speaker-recognition-calibration-panel"' in frontend
    assert "speaker-recognition-calibration-panel.js" in frontend
    assert "More labelled decisions needed" in panel
    assert "Apply suggested threshold" in panel
    assert "apply_recommended_threshold" in panel
    assert "Changing the HA threshold cannot fix" in panel
    assert "wrong person ×" in panel


def test_changed_python_modules_compile() -> None:
    for name in ("calibration.py", "telemetry.py", "websocket.py", "frontend.py"):
        source = (ROOT / name).read_text(encoding="utf-8")
        compile(source, str(ROOT / name), "exec")
