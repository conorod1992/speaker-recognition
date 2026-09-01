"""Contract coverage for the experimental engine-comparison user interface."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
PANEL = ROOT / "www" / "speaker-recognition-evaluation-panel.js"
FRONTEND = ROOT / "frontend.py"
SHADOW_WEBSOCKET = ROOT / "shadow_websocket.py"


def test_main_panel_loads_evaluation_wrapper_without_renaming_element() -> None:
    """The wrapper augments the established sidebar custom element in place."""
    frontend = FRONTEND.read_text(encoding="utf-8")
    panel = PANEL.read_text(encoding="utf-8")

    assert 'PANEL_ELEMENT = "speaker-recognition-settings-panel"' in frontend
    assert 'BASE_PANEL_MODULE = "speaker-recognition-settings-panel.js"' in frontend
    assert 'module_url=f"{STATIC_URL}/speaker-recognition-evaluation-panel.js"' in frontend
    assert 'import "./speaker-recognition-settings-panel.js"' in panel
    assert 'customElements.get("speaker-recognition-settings-panel")' in panel
    assert "proto._organizePanel = function()" in panel
    assert 'customElements.define("speaker-recognition-evaluation-panel"' not in panel


def test_evaluation_tab_surfaces_live_paired_model_evidence() -> None:
    """Users get a dedicated live A/B flow with independent ground truth and metrics."""
    source = PANEL.read_text(encoding="utf-8")

    assert 'type: "speaker_recognition/evaluation_status"' in source
    assert 'button.dataset.panelTab = "evaluation"' in source
    assert "Start testing" in source
    assert "Was that" in source
    assert "someone not enrolled" in source
    assert "Discard trial" in source
    assert "Clear results" in source
    assert "Correct decisions" in source
    assert "Wrong-speaker / false accepts" in source
    assert "False unknowns" in source
    assert "Best similarity threshold" in source
    assert "Best margin threshold" in source
    assert "Median backend time" in source
    assert "Median end-to-end model call" in source
    assert "Median effective Assist latency" in source
    assert "raw similarity values are not comparable between models" in source


def test_evaluation_ui_keeps_shadow_engine_non_authoritative() -> None:
    """The experiment is clearly described as unable to alter HA identity."""
    source = PANEL.read_text(encoding="utf-8")

    assert "Resemblyzer remains authoritative." in source
    assert "ECAPA runs beside it only for measurement" in source
    assert "never changes the identity Home Assistant uses" in source
    assert "Preparing profiles" in source
    assert "Ground truth needed" in source
    assert "Scoring utterance" in source
    assert "Waiting for utterance" in source


def test_evaluation_websocket_reports_readiness_and_live_controls() -> None:
    """Live evaluation is admin-controlled while retaining the legacy endpoint."""
    source = SHADOW_WEBSOCKET.read_text(encoding="utf-8")

    assert 'f"{DOMAIN}/shadow_comparison"' in source
    assert 'f"{DOMAIN}/evaluation_status"' in source
    assert 'f"{DOMAIN}/evaluation_start"' in source
    assert 'f"{DOMAIN}/evaluation_stop"' in source
    assert 'f"{DOMAIN}/evaluation_label"' in source
    assert 'f"{DOMAIN}/evaluation_discard"' in source
    assert 'f"{DOMAIN}/evaluation_clear"' in source
    assert '"enabled": recognition.shadow_engine_id is not None' in source
    assert '"ready": recognition.shadow_ready' in source
    assert '"configured_users": sorted(recognition.configured_users)' in source
    assert '"enrolled_users": sorted(recognition.shadow_enrolled_users)' in source
