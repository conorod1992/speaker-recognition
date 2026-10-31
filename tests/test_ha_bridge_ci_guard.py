"""Regression tests for CI's real add-on acceptance result gate."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.assert_ha_bridge_results import REQUIRED_TESTS, verify_results


def _write_report(tmp_path: Path, statuses: dict[str, str] | None = None) -> Path:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", name="pytest")
    for name in sorted(REQUIRED_TESTS):
        case = ET.SubElement(
            suite, "testcase", classname="tests_real_ha.test_real_addon_bridge",
            name=name,
        )
        status = (statuses or {}).get(name)
        if status:
            ET.SubElement(case, status, message=status)
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path)
    return path


def test_accepts_two_executed_successful_cases(tmp_path: Path) -> None:
    verify_results(_write_report(tmp_path))


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_rejects_any_required_case_not_passing(tmp_path: Path, outcome: str) -> None:
    name = sorted(REQUIRED_TESTS)[0]
    with pytest.raises(RuntimeError, match="did not pass"):
        verify_results(_write_report(tmp_path, {name: outcome}))


def test_rejects_missing_required_case(tmp_path: Path) -> None:
    path = _write_report(tmp_path)
    tree = ET.parse(path)
    case = next(tree.iter("testcase"))
    parent = next(tree.iter("testsuite"))
    parent.remove(case)
    tree.write(path)
    with pytest.raises(RuntimeError, match="not collected"):
        verify_results(path)


def test_rejects_only_unrelated_tests(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite><testcase classname="other.tests" name="test_pass"/></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not collected"):
        verify_results(path)


def test_rejects_missing_report(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        verify_results(tmp_path / "missing.xml")


def test_rejects_malformed_report(tmp_path: Path) -> None:
    path = tmp_path / "invalid.xml"
    path.write_text("<testsuite>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        verify_results(path)


def test_rejects_duplicate_cases(tmp_path: Path) -> None:
    path = _write_report(tmp_path)
    tree = ET.parse(path)
    suite = next(tree.iter("testsuite"))
    ET.SubElement(
        suite, "testcase",
        classname="tests_real_ha.test_real_addon_bridge",
        name=sorted(REQUIRED_TESTS)[0],
    )
    tree.write(path)
    with pytest.raises(RuntimeError, match="Duplicate"):
        verify_results(path)


def test_bridge_workflow_requires_junit_assertion() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/real-ha-addon-bridge.yml"
    ).read_text(encoding="utf-8")
    assert "--junitxml=" in workflow
    assert "scripts/assert_ha_bridge_results.py" in workflow
    assert "python -m pip check" in workflow


def test_stable_ha_workflow_does_not_override_harness_pin() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/real-ha.yml"
    ).read_text(encoding="utf-8")
    assert "python -m pip install --upgrade homeassistant" not in workflow
    assert "python -m pip check" in workflow


def test_rejects_newly_added_skipped_bridge_case(tmp_path: Path) -> None:
    path = _write_report(tmp_path)
    tree = ET.parse(path)
    suite = next(tree.iter("testsuite"))
    extra = ET.SubElement(
        suite, "testcase",
        classname="tests_real_ha.test_real_addon_bridge",
        name="test_future_addon_workflow",
    )
    ET.SubElement(extra, "skipped", message="unexpected skip")
    tree.write(path)
    with pytest.raises(RuntimeError, match="test_future_addon_workflow.*did not pass"):
        verify_results(path)


def test_accepts_additional_successful_bridge_case(tmp_path: Path) -> None:
    path = _write_report(tmp_path)
    tree = ET.parse(path)
    suite = next(tree.iter("testsuite"))
    ET.SubElement(
        suite, "testcase",
        classname="tests_real_ha.test_real_addon_bridge",
        name="test_future_addon_workflow",
    )
    tree.write(path)
    verify_results(path)


def test_latest_stable_probe_is_not_part_of_normal_pr_lane() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/real-ha.yml"
    ).read_text(encoding="utf-8")
    assert "Install latest published stable HA (compatibility probe)" in workflow
    assert "Probe latest-stable Home Assistant runtime" in workflow
    expected = (
        "matrix.home-assistant == 'stable' && "
        "(github.event_name == 'schedule' || github.event_name == 'workflow_dispatch')"
    )
    assert workflow.count("if: " + expected) == 2
