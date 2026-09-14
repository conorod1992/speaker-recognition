"""Shared fixtures for tests that run against a real Home Assistant runtime."""

from __future__ import annotations

from collections.abc import Generator

import pytest

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(
    enable_custom_integrations: None,
) -> Generator[None, None, None]:
    """Allow Home Assistant to load this repository's custom integration."""
    yield
