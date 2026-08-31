"""Contracts for reproducible release metadata."""

from pathlib import Path
import re

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


def test_release_version_is_2_9_0() -> None:
    """The post-sweep release is consistently staged as 2.9.0."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.9.0"' in pyproject
