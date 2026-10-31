"""Fail CI unless all required real HA-to-add-on acceptance tests actually ran."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

REQUIRED_TESTS = {
    "test_real_ha_main_entry_trains_recognizes_and_reconciles_real_addon",
    "test_real_ha_wrong_addon_token_fails_setup_closed",
}


def verify_results(path: Path) -> None:
    root = ET.parse(path).getroot()
    cases = [
        case
        for case in root.iter("testcase")
        if case.get("classname", "").endswith("test_real_addon_bridge")
    ]
    by_name = {case.get("name"): case for case in cases}
    if len(by_name) != len(cases):
        raise RuntimeError("Duplicate real add-on bridge JUnit test cases")
    missing = REQUIRED_TESTS - by_name.keys()
    if missing:
        raise RuntimeError(f"Real add-on bridge tests not collected: {sorted(missing)}")
    # Every collected bridge test must run, including tests added in the future.
    for name, case in sorted(by_name.items()):
        outcomes = [
            child.tag for child in case if child.tag in {"skipped", "failure", "error"}
        ]
        if outcomes:
            raise RuntimeError(f"Real add-on bridge test {name} did not pass: {outcomes}")
    print(f"Verified all {len(cases)} real add-on bridge tests executed and passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml", type=Path)
    verify_results(parser.parse_args().junit_xml)
