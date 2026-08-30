"""Source-level contracts for HA-bound robustness paths."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / "speaker_recognition"


def test_satellite_claims_require_unambiguous_listening_origin() -> None:
    enrollment = (INTEGRATION / "enrollment.py").read_text(encoding="utf-8")
    diagnostics = (INTEGRATION / "diagnostics.py").read_text(encoding="utf-8")

    assert 'hass.states.async_all("assist_satellite")' in enrollment
    assert "if len(listening) != 1:" in enrollment
    assert 'hass.states.async_all("assist_satellite")' in diagnostics
    assert "if satellite_id != session.satellite_id:" in diagnostics
    assert "elif not claimed_match:" in diagnostics


def test_release_assets_use_builtin_github_token() -> None:
    publish = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    assert "secrets.PAT" not in publish
    assert "actions/upload-release-asset" not in publish
    assert "GH_TOKEN: ${{ github.token }}" in publish
    assert "gh release upload" in publish


def test_standalone_backend_python_support_matches_ml_dependencies() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = "==3.9.*"' in pyproject
    assert "python_version < '3.10'" not in pyproject


def test_frontend_polling_does_not_reschedule_while_detached() -> None:
    panel = (
        INTEGRATION / "www" / "speaker-recognition-settings-panel.js"
    ).read_text(encoding="utf-8")

    assert "if (!this.isConnected || !this._status) return;" in panel
    assert panel.count("if (!this.isConnected) return;") >= 3
    assert "if (this.isConnected) this._pollTimer = setTimeout" in panel
    assert "if (this.isConnected) this._livePollTimer = setTimeout" in panel
