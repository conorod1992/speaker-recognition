"""Regression tests for Home Assistant integration resources."""

import json
from pathlib import Path


def test_config_menu_has_current_helper_labels() -> None:
    """The config flow menu options must have matching translation keys."""
    translations_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "translations"
        / "en.json"
    )
    translations = json.loads(translations_path.read_text(encoding="utf-8"))

    menu_options = translations["config"]["step"]["menu"]["menu_options"]
    assert menu_options == {
        "add_stt": "Add STT proxy",
        "add_conversation": "Add Conversation proxy",
    }


def test_guided_enrollment_steps_have_config_and_options_translations() -> None:
    """Every shared guided step must render in initial and retraining flows."""
    translations_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "translations"
        / "en.json"
    )
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    guided_steps = {
        "enrollment_user",
        "enrollment_sample",
        "enrollment_review",
        "enrollment_complete",
    }

    assert guided_steps <= translations["config"]["step"].keys()
    assert guided_steps <= translations["options"]["step"].keys()
    sample_description = translations["options"]["step"]["enrollment_sample"][
        "description"
    ]
    assert "{phrase}" in sample_description
    assert "{accepted}" in sample_description


def test_media_source_dependency_and_addon_architectures_are_preserved() -> None:
    """Enrollment media resolves natively and both supported images stay enabled."""
    root = Path(__file__).parents[1]
    manifest = json.loads(
        (
            root / "custom_components" / "speaker_recognition" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    addon_config = (root / "speaker_recognition_addon" / "config.yaml").read_text(
        encoding="utf-8"
    )

    assert "media_source" in manifest["dependencies"]
    assert "  - amd64" in addon_config
    assert "  - aarch64" in addon_config
