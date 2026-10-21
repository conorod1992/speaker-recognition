"""Run pytest after Home Assistant has initialized its dependency aliases."""

import sys


def _prepare_ha_logger_test_compatibility() -> None:
    """Bridge the logger helper move until the HA pytest harness catches up.

    The published pytest-homeassistant-custom-component harness can target the
    latest stable HA while this workflow also exercises the upcoming HA dev
    branch. HA dev moved _clear_logger_overwrites out of the logger package
    into logger.helpers, but the stable harness still imports the old path.
    Only restore that existing helper's import path when it is missing.
    """
    from homeassistant.components import logger

    if not hasattr(logger, "_clear_logger_overwrites"):
        from homeassistant.components.logger.helpers import _clear_logger_overwrites

        logger._clear_logger_overwrites = _clear_logger_overwrites


def main() -> int:
    """Initialize HA and bridge its logger helper before pytest plugin loading."""
    # HA dev aliases httpx/httpcore in its package initializer. The HA pytest
    # harness and respx plugin import httpx before conftest.py is loaded, so a
    # fixture or conftest import is too late. Let HA own the initialization,
    # just as upstream tests/__init__.py does; stable HA also works unchanged.
    import homeassistant  # noqa: F401

    _prepare_ha_logger_test_compatibility()

    import pytest

    return pytest.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
