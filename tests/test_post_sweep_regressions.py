"""Regression coverage for issues found by the post-merge robustness sweep."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from speaker_recognition.models import Config

ROOT = Path(__file__).parents[1]
BACKEND = ROOT / "speaker_recognition"
ADDON_BACKEND = ROOT / "speaker_recognition_addon" / "speaker_recognition"


def test_runtime_config_validates_assignment() -> None:
    """Pydantic v2 must validate mutations of the process-wide runtime config."""
    runtime_config = Config()

    with pytest.raises(ValidationError):
        runtime_config.port = "not-a-port"  # type: ignore[assignment]

    assert runtime_config.port == 8099


def test_backend_model_copy_stays_synchronized() -> None:
    """The Supervisor add-on must ship the same corrected Pydantic model."""
    assert (BACKEND / "models.py").read_text(encoding="utf-8") == (
        ADDON_BACKEND / "models.py"
    ).read_text(encoding="utf-8")
