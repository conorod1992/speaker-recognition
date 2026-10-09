"""Run pytest after Home Assistant has initialized its dependency aliases."""

import sys


def main() -> int:
    """Initialize HA before pytest auto-loads plugins such as respx."""
    # HA dev aliases httpx/httpcore in its package initializer. The HA pytest
    # harness and respx plugin import httpx before conftest.py is loaded, so a
    # fixture or conftest import is too late. Let HA own the initialization,
    # just as upstream tests/__init__.py does; stable HA also works unchanged.
    import homeassistant  # noqa: F401
    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
