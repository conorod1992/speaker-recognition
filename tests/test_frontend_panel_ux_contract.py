"""Contract coverage for the sidebar panel information architecture and scanability."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
PANEL = ROOT / "www" / "speaker-recognition-settings-panel.js"


def _source() -> str:
    return PANEL.read_text(encoding="utf-8")


def test_panel_separates_major_tasks_into_tabs() -> None:
    """Enrollment, diagnostics, calibration and settings are separate surfaces."""
    source = _source()

    assert 'this._panelSection = "enrollment"' in source
    assert '["enrollment", "Enrollment"]' in source
    assert '["diagnostics", "Diagnostics"]' in source
    assert '["calibration", "Calibration"]' in source
    assert '["settings", "Settings"]' in source
    assert 'tabs.setAttribute("role", "tablist")' in source
    assert 'data-panel-section' in source


def test_enrollment_surface_has_compact_progress_and_selected_step() -> None:
    """The primary enrollment workflow exposes progress without four large status tiles."""
    source = _source()

    assert "profileSummaryChips" in source
    assert "Training samples" in source
    assert 'button.classList.toggle("active", index === this._sampleIndex)' in source
    assert 'aria-current' in source
    assert "more sample" in source
    assert 'commit.textContent = enrolled ? "Retrain profile" : "Train profile"' in source


def test_user_facing_diagnostics_use_ha_names_not_raw_ids() -> None:
    """Normal UI resolves configured IDs through the Home Assistant user list."""
    source = _source()

    assert "this._status.users.find(item => item.id === userId)" in source
    assert "Unknown HA user" in source
    assert "this._userName(result.candidate_user_id)" in source
    assert "names = enrolled.map(userId => this._userName(userId))" in source


def test_panel_adds_responsive_and_semantic_visual_hierarchy() -> None:
    """The refresh includes narrow-screen controls and distinct result states."""
    source = _source()

    assert "@media (max-width: 700px)" in source
    assert ".panelTab.active" in source
    assert ".message.message-success" in source
    assert ".message.message-error" in source
    assert ".trainingAction" in source
