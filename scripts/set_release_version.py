#!/usr/bin/env python3
"""Validate and stage a Speaker Recognition release version.

Normal development keeps every committed version-bearing file at the most recent
published release. The release workflow calls this script in isolated checkouts
to build the next version before exposing that version through the add-on
repository metadata on ``master``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re

ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


@dataclass(frozen=True)
class VersionSpec:
    """One version-bearing file and the patterns used to read/update it."""

    label: str
    path: str
    extract_pattern: str
    replace_pattern: str
    flags: int = 0


SPECS = (
    VersionSpec(
        "package",
        "pyproject.toml",
        r'^version = "([^"]+)"$',
        r'(^version = ")[^"]+("$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "Home Assistant integration",
        "custom_components/speaker_recognition/manifest.json",
        r'^  "version": "([^"]+)"$',
        r'(^  "version": ")[^"]+("$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "backend API",
        "speaker_recognition/api.py",
        r'^    version="([^"]+)",$',
        r'(^    version=")[^"]+(",$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "add-on metadata",
        "speaker_recognition_addon/config.yaml",
        r'^version: "([^"]+)"$',
        r'(^version: ")[^"]+("$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "add-on Docker default",
        "speaker_recognition_addon/Dockerfile",
        r'^ARG BUILD_VERSION="([^"]+)"$',
        r'(^ARG BUILD_VERSION=")[^"]+("$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "add-on API",
        "speaker_recognition_addon/speaker_recognition/api.py",
        r'^    version="([^"]+)",$',
        r'(^    version=")[^"]+(",$)',
        re.MULTILINE,
    ),
    VersionSpec(
        "lockfile package",
        "uv.lock",
        r'^\[\[package\]\]\nname = "hass-speaker-recognition"\nversion = "([^"]+)"',
        r'(^\[\[package\]\]\nname = "hass-speaker-recognition"\nversion = ")[^"]+("$)',
        re.MULTILINE,
    ),
)


def _match_once(root: Path, spec: VersionSpec) -> tuple[str, str]:
    path = root / spec.path
    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(spec.extract_pattern, text, spec.flags))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {spec.label} version in {spec.path}; "
            f"found {len(matches)}"
        )
    return text, matches[0].group(1)


def read_versions(root: Path = ROOT) -> dict[str, str]:
    """Return all tracked release versions, failing on ambiguous file shapes."""
    return {spec.label: _match_once(root, spec)[1] for spec in SPECS}


def current_version(root: Path = ROOT) -> str:
    """Return the common tracked version, failing if files disagree."""
    versions = read_versions(root)
    unique = set(versions.values())
    if len(unique) != 1:
        raise RuntimeError(f"Release versions are inconsistent: {versions}")
    return next(iter(unique))


def set_release_version(version: str, root: Path = ROOT) -> None:
    """Set every tracked release-version field to ``version``."""
    if not SEMVER.fullmatch(version):
        raise ValueError(f"Release version must use X.Y.Z format, got {version!r}")

    # Validate all source files before mutating any of them.
    current_version(root)

    for spec in SPECS:
        path = root / spec.path
        text = path.read_text(encoding="utf-8")
        replacement = lambda match: f"{match.group(1)}{version}{match.group(2)}"
        updated, count = re.subn(
            spec.replace_pattern,
            replacement,
            text,
            count=1,
            flags=spec.flags,
        )
        if count != 1:
            raise RuntimeError(
                f"Expected exactly one writable {spec.label} version in {spec.path}; "
                f"found {count}"
            )
        path.write_text(updated, encoding="utf-8")

    staged = current_version(root)
    if staged != version:
        raise RuntimeError(f"Version staging produced {staged!r}, expected {version!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version", nargs="?", help="target X.Y.Z release version")
    parser.add_argument(
        "--check",
        action="store_true",
        help="only verify that all tracked version fields agree",
    )
    args = parser.parse_args()

    if args.check:
        if args.version is not None:
            parser.error("--check does not accept a target version")
        print(current_version())
        return
    if args.version is None:
        parser.error("a target version is required unless --check is used")

    before = current_version()
    set_release_version(args.version)
    print(f"Staged release version {before} -> {args.version}")


if __name__ == "__main__":
    main()
