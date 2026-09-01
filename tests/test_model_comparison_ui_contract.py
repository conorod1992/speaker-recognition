"""Contract coverage for the experimental engine-comparison user interface."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
PANEL = ROOT / "www" / "speaker-recognition-evaluation-panel.js"
FRONTEND = ROOT / "frontend.py"
SHADOW_WEBSOCKET = ROOT / "shadow_websocket.py"


def test_main_panel_registers_evaluation_wrapper() -> None:
    """The sidebar loads the wrapper that extends the existing settings panel."""
    source = FRONTEND.read_text(encoding="utf-8")

    assert 'PANEL_ELEMENT = "speaker-recognition-evaluation-panel"' in source
    assert 'module_url=f"{STATIC_URL}/speaker-recognition-evaluation-panel.js"' in source


def test_evaluation_tab_surfaces_paired_model_evidence() -> None:
    """Users can see progress, independent operating points, errors and latency."""
    source = PANEL.read_text(encoding="utf-8")

    assert 'type: "speaker_recognition/shadow_comparison"' in source
    assert 'button.dataset.panelTab = "evaluation"' in source
    assert "paired labelled decisions" in source
    assert "Correct decisions" in source
    assert "Wrong-speaker / false accepts" in source
    assert "False unknowns" in source
    assert "Best similarity threshold" in source
    assert "Best margin threshold" in source
    assert "Median backend time" in source
    assert "raw similarity values are not directly comparable between models" in source


def test_evaluation_ui_keeps_shadow_engine_non_authoritative() -> None:
    """The experiment is clearly described as unable to alter HA identity."""
    source = PANEL.read_text(encoding="utf-8")

    assert "Resemblyzer remains authoritative." in source
    assert "ECAPA results are experimental only" in source
    assert "Preparing profiles" in source
    assert "Sufficient evidence" in source
    assert "Collecting evidence" in source


def test_shadow_comparison_reports_live_engine_readiness() -> None:
    """The comparison endpoint is useful before paired labelled trials exist."""
    source = SHADOW_WEBSOCKET.read_text(encoding="utf-8")

    assert 'result["shadow_status"] = _shadow_status(hass)' in source
    assert '"enabled": recognition.shadow_engine_id is not None' in source
    assert '"ready": recognition.shadow_ready' in source
    assert '"configured_users": sorted(recognition.configured_users)' in source
    assert '"enrolled_users": sorted(recognition.shadow_enrolled_users)' in source
