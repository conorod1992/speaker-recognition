"""Contract coverage for the sidebar panel information architecture and scanability."""

from pathlib import Path

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
PANEL = ROOT / "www" / "speaker-recognition-settings-panel.js"


def _source() -> str:
    return PANEL.read_text(encoding="utf-8")


def test_panel_separates_user_tasks_into_plain_language_tabs() -> None:
    """The normal panel is organised around user goals rather than internals."""
    source = _source()

    assert 'this._panelSection = "enrollment"' in source
    assert '["enrollment", "Voices"]' in source
    assert '["recognition", "Recognition"]' in source
    assert '["improve", "Improve accuracy"]' in source
    assert '["settings", "Settings"]' in source
    assert '["diagnostics", "Diagnostics"]' not in source
    assert '["calibration", "Calibration"]' not in source
    assert 'tabs.setAttribute("role", "tablist")' in source
    assert "data-panel-section" in source


def test_enrollment_surface_explains_active_profile_and_progress() -> None:
    """Enrollment distinguishes an active profile from an update in progress."""
    source = _source()

    assert "✓ Voice profile active" in source
    assert "Updating profile" in source
    assert "Training phrases" in source
    assert "of ${minimum} needed" in source
    assert "Record at least ${minimum} of the ${total} phrases below." in source
    assert 'button.classList.toggle("active", index === this._sampleIndex)' in source
    assert "aria-current" in source
    assert 'commit.textContent = enrolled ? "Update voice profile" : "Create voice profile"' in source
    assert ">Retraining<" not in source


def test_user_facing_diagnostics_use_ha_names_and_hide_raw_metrics() -> None:
    """Normal UI resolves names and keeps model metrics behind technical details."""
    source = _source()

    assert "this._status.users.find(item => item.id === userId)" in source
    assert "Unknown HA user" in source
    assert "this._userName(result.candidate_user_id)" in source
    assert "names = enrolled.map(userId => ({ userId, name: this._userName(userId) }))" in source
    assert "<summary>Technical details</summary>" in source
    assert "No comparison needed yet" in source
    assert "No unusually similar enrolled voice was found." in source
    assert "data-profile-health-user" in source
    assert 'class="profileName ${item.userId === this._profileHealthUserId ? "selected" : ""}"' in source
    assert "This measures how similar this voice's training recordings are to one another." in source
    assert ".profileHealthActions { margin-top:18px; }" in source


def test_settings_keep_tuning_controls_advanced() -> None:
    """Technical thresholds remain available without dominating normal settings."""
    source = _source()

    assert "Advanced recognition settings" in source
    assert "Recognition strictness" in source
    assert "Ambiguous-match protection" in source
    assert "Advanced identity setting" in source
    assert "Automatic recommendations are available under Improve accuracy." in source
    assert "#calibrationGuidanceCard { background:var(--card-background-color) !important; }" in source


def test_panel_adds_responsive_and_semantic_visual_hierarchy() -> None:
    """The refresh includes narrow-screen controls and distinct result states."""
    source = _source()

    assert "@media (max-width: 700px)" in source
    assert ".panelTab.active" in source
    assert ".message.message-success" in source
    assert ".message.message-error" in source
    assert ".trainingAction" in source
