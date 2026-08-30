"""Regression contracts for proxy source configuration and cycle prevention."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / "speaker_recognition"


def _source(name: str) -> str:
    return (INTEGRATION / name).read_text(encoding="utf-8")


def test_conversation_proxy_uses_effective_options() -> None:
    """Conversation source and threshold must honor saved options after reload."""
    source = _source("conversation.py")
    assert "config_entry.options.get(\n        CONF_CONVERSATION_ENTITY" in source
    assert "def min_confidence" in source
    assert "self._config_entry.options.get(\n                CONF_MIN_CONFIDENCE" in source
    assert "min_confidence = self.min_confidence" in source


def test_options_forms_reopen_with_effective_values() -> None:
    """Reopening options must not silently restore original data values."""
    source = _source("config_flow.py")
    assert "current_conversation_entity = self.config_entry.options.get(" in source
    assert "current_min_confidence = self.config_entry.options.get(" in source
    assert "current_stt_entity = self.config_entry.options.get(" in source


def test_proxy_sources_are_validated_in_add_and_options_flows() -> None:
    """Both creation and editing reject duplicate or recursive proxy sources."""
    source = _source("config_flow.py")
    assert source.count("validate_proxy_source(") >= 4
    assert "exclude_entry_id=self.config_entry.entry_id" in source
    helper = _source("proxy.py")
    assert "is_speaker_recognition_proxy_entity" in helper
    assert "proxy_source_in_use" in helper
    assert 'return "speaker_recognition_proxy_source"' in helper
    assert 'return "proxy_source_already_wrapped"' in helper


def test_persisted_proxy_entries_are_revalidated_and_unique_ids_repaired() -> None:
    """Existing stale options cannot bypass validation after Home Assistant restarts."""
    source = _source("__init__.py")
    assert "def _prepare_proxy_entry" in source
    assert "validate_proxy_source(" in source
    assert "sync_proxy_unique_id(hass, entry, entry_type)" in source
    assert "_prepare_proxy_entry(hass, entry, ENTRY_TYPE_STT)" in source
    assert "_prepare_proxy_entry(hass, entry, ENTRY_TYPE_CONVERSATION)" in source
