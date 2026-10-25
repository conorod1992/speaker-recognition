"""Protect stable/upcoming Home Assistant interpreter selection."""

from unittest.mock import patch

import pytest

from scripts.resolve_ha_python import harness_core_version, latest_harness_version, resolve_python


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


def test_resolver_uses_versioned_harness_metadata() -> None:
    with patch("scripts.resolve_ha_python._get_json", return_value={
        "info": {"requires_dist": ["homeassistant==2026.10.2"]}
    }) as request, patch(
        "scripts.resolve_ha_python._get_text", return_value="3.14.9"
    ):
        resolve_python("stable", harness_version="0.13.373")
    assert request.call_args.args[0].endswith("/0.13.373/json")


def test_harness_release_version_is_validated() -> None:
    with patch("scripts.resolve_ha_python._get_json", return_value={
        "info": {"version": "0.13.373"}
    }):
        assert latest_harness_version() == "0.13.373"

    with patch("scripts.resolve_ha_python._get_json", return_value={
        "info": {"version": "not-a-version"}
    }):
        with pytest.raises(ValueError, match="Unexpected"):
            latest_harness_version()


def test_real_ha_workflows_pin_selected_harness() -> None:
    from pathlib import Path

    root = Path(__file__).parents[1] / ".github/workflows"
    for name in ("real-ha.yml", "real-ha-addon-bridge.yml"):
        workflow = (root / name).read_text(encoding="utf-8")
        assert "resolve_ha_python.py harness-version" in workflow
        assert "resolve_ha_python.py stable --harness-version" in workflow
        assert "pytest-homeassistant-custom-component==${{" in workflow
