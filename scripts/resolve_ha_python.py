"""Resolve Python independently for the HA harness and upcoming HA Core."""

from __future__ import annotations

import argparse
import json
import re
from urllib.request import urlopen

CORE_RAW = "https://raw.githubusercontent.com/home-assistant/core"
PYPI_HARNESS = "https://pypi.org/pypi/pytest-homeassistant-custom-component/json"


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=20) as response:
        return json.load(response)


def _get_text(url: str) -> str:
    with urlopen(url, timeout=20) as response:
        return response.read().decode("utf-8").strip()


def harness_core_version(requires_dist: list[str]) -> str:
    """Find the exact HA release selected by the published test harness."""
    versions = set()
    for requirement in requires_dist:
        match = re.match(r"^homeassistant\s*==\s*([0-9]+(?:\.[0-9]+){2})\b", requirement, re.I)
        if match:
            versions.add(match.group(1))
    if len(versions) != 1:
        raise ValueError(f"Expected exactly one pinned HA Core requirement, got {sorted(versions)}")
    return versions.pop()


def resolve_python(lane: str) -> tuple[str, str]:
    if lane == "upcoming":
        ref = "dev"
    elif lane == "stable":
        harness = _get_json(PYPI_HARNESS)
        ref = harness_core_version(harness["info"]["requires_dist"])
    else:
        raise ValueError(f"Unknown HA lane: {lane}")

    version = _get_text(f"{CORE_RAW}/{ref}/.python-version")
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", version):
        raise ValueError(f"Invalid Python version from HA Core {ref}: {version!r}")
    return ref, version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane", choices=("stable", "upcoming"))
    args = parser.parse_args()
    ref, python_version = resolve_python(args.lane)
    print(f"Home Assistant {args.lane}: Core {ref}; Python {python_version}")
    print(python_version, end="")


if __name__ == "__main__":
    main()
