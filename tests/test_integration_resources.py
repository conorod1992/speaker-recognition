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
