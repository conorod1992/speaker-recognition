"""Contracts for reproducible release metadata."""

from pathlib import Path
import re
import shutil

from scripts.set_release_version import SPECS, current_version, read_versions, set_release_version

ROOT = Path(__file__).parents[1]


def test_uv_lock_python_range_matches_pyproject() -> None:
    """The committed lock must reflect the package's supported Python range."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    project_range = re.search(r'(?m)^requires-python = "([^"]+)"$', pyproject)
    lock_range = re.search(r'(?m)^requires-python = "([^"]+)"$', lock)
    assert project_range is not None
    assert lock_range is not None
    assert project_range.group(1).replace(" ", "") == lock_range.group(1).replace(" ", "")


def test_committed_release_versions_are_consistent() -> None:
    """Master keeps every version-bearing file on one released version."""
    versions = read_versions(ROOT)
    assert len(set(versions.values())) == 1
    assert current_version(ROOT) == next(iter(versions.values()))


def test_release_version_stager_updates_every_tracked_file(tmp_path: Path) -> None:
    """The release workflow can stage the next version without a manual bump PR."""
    for spec in SPECS:
        source = ROOT / spec.path
        destination = tmp_path / spec.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    set_release_version("9.8.7", tmp_path)

    assert current_version(tmp_path) == "9.8.7"
    assert set(read_versions(tmp_path).values()) == {"9.8.7"}
