"""Synchronize the backend package into the Supervisor build context."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPOSITORY_ROOT / "speaker_recognition"
DESTINATION = REPOSITORY_ROOT / "speaker_recognition_addon" / "speaker_recognition"
IGNORED_NAMES = {"__pycache__"}
TEXT_SUFFIXES = {".py", ".typed"}


def source_files(root: Path) -> set[Path]:
    """Return package files relative to *root*."""
    return {
        path.relative_to(root)
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_NAMES for part in path.parts)
        and path.suffix != ".pyc"
    }


def differences() -> list[str]:
    """Describe files missing, extra, or changed in the add-on copy."""
    expected = source_files(SOURCE)
    actual = source_files(DESTINATION) if DESTINATION.exists() else set()
    result = [f"missing: {path}" for path in sorted(expected - actual)]
    result.extend(f"extra: {path}" for path in sorted(actual - expected))
    result.extend(
        f"changed: {path}"
        for path in sorted(expected & actual)
        if not files_match(path)
    )
    return result


def files_match(relative_path: Path) -> bool:
    """Compare source files while ignoring platform newline conversion."""
    source_data = (SOURCE / relative_path).read_bytes()
    destination_data = (DESTINATION / relative_path).read_bytes()
    if relative_path.suffix in TEXT_SUFFIXES:
        source_data = source_data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        destination_data = destination_data.replace(b"\r\n", b"\n").replace(
            b"\r", b"\n"
        )
    return source_data == destination_data


def synchronize() -> None:
    """Replace the add-on copy with the canonical backend package."""
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    shutil.copytree(
        SOURCE,
        DESTINATION,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def main() -> int:
    """Synchronize sources or verify that the committed copy is current."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without modifying the add-on copy",
    )
    args = parser.parse_args()

    if not args.check:
        synchronize()

    drift = differences()
    if drift:
        print("Add-on backend sources are out of sync:", file=sys.stderr)
        for item in drift:
            print(f"  {item}", file=sys.stderr)
        print(
            "Run: python scripts/sync_addon_sources.py",
            file=sys.stderr,
        )
        return 1

    print("Add-on backend sources match the repository package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
