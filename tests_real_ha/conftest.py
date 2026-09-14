"""Shared fixtures for tests that run against a real Home Assistant runtime."""

from __future__ import annotations

from collections.abc import Generator

import pytest

pytest_plugins = "pytest_homeassistant_custom_component"


@pytest.fixture(autouse=True)
def real_ha_acceptance_environment(
    enable_custom_integrations: None,
    socket_enabled: None,
) -> Generator[None, None, None]:
    """Enable custom integrations and local HTTP sockets for Real HA acceptance tests.

    These tests deliberately run a loopback aiohttp backend so the integration's
    real Home Assistant -> aiohttp -> backend boundary is exercised. The HA test
    harness blocks sockets by default, so its ``socket_enabled`` fixture is
    required for this dedicated acceptance-test suite.
    """
    yield
