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
