"""Protect stable/upcoming Home Assistant interpreter selection."""

from unittest.mock import patch

import pytest

from scripts.resolve_ha_python import harness_core_version, resolve_python


def test_extracts_exact_home_assistant_pin() -> None:
    assert harness_core_version([
        "pytest>=8",
        "homeassistant==2026.10.2",
        "aiohttp>=3",
    ]) == "2026.10.2"


@pytest.mark.parametrize("requirements", [
    [],
    ["homeassistant>=2026.10.2"],
    ["homeassistant==2026.10.2", "homeassistant==2026.10.3"],
])
def test_rejects_missing_or_ambiguous_pin(requirements: list[str]) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        harness_core_version(requirements)


def test_stable_uses_pinned_core_release_python() -> None:
    with patch("scripts.resolve_ha_python._get_json", return_value={
        "info": {"requires_dist": ["homeassistant==2026.10.2"]}
    }), patch("scripts.resolve_ha_python._get_text", return_value="3.14.9") as read:
        assert resolve_python("stable") == ("2026.10.2", "3.14.9")
    assert "/2026.10.2/.python-version" in read.call_args.args[0]


def test_upcoming_uses_dev_python_without_pypi_lookup() -> None:
    with patch("scripts.resolve_ha_python._get_json") as pypi, patch(
        "scripts.resolve_ha_python._get_text", return_value="3.15"
    ) as read:
        assert resolve_python("upcoming") == ("dev", "3.15")
    pypi.assert_not_called()
    assert "/dev/.python-version" in read.call_args.args[0]


def test_bad_python_version_fails_closed() -> None:
    with patch("scripts.resolve_ha_python._get_text", return_value="malformed"):
        with pytest.raises(ValueError, match="Invalid Python version"):
            resolve_python("upcoming")


def test_unknown_lane_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        resolve_python("beta")
