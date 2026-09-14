"""Real Home Assistant acceptance tests for the main config-entry lifecycle."""

from __future__ import annotations

import socket
from typing import Any

from aiohttp import web
import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)


class BackendStub:
    """Small real HTTP backend used by the HA acceptance tests."""

    def __init__(self, port: int = 0) -> None:
        self.port = port
        self.health_status = 200
        self.health_payload: Any = {
            "status": "healthy",
            "trained": False,
            "enrolled_users": [],
            "encoder_ready": True,
            "warmup_error": None,
        }
        self.requests: list[tuple[str, str | None]] = []
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/profiles/sync", self._sync_profiles)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        sockets = site._server.sockets if site._server is not None else []  # noqa: SLF001
        assert sockets
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _health(self, request: web.Request) -> web.Response:
        self.requests.append(("GET /health", request.headers.get("Authorization")))
        return web.json_response(self.health_payload, status=self.health_status)

    async def _sync_profiles(self, request: web.Request) -> web.Response:
        self.requests.append(
            ("POST /profiles/sync", request.headers.get("Authorization"))
        )
        payload = await request.json()
        desired_users = payload.get("desired_users", [])
        return web.json_response(
            {"enrolled_users": desired_users, "removed_users": []}
        )


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _main_entry(backend_url: str, *, token: str = "") -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: backend_url,
            CONF_BACKEND_TOKEN: token,
        },
        options={CONF_VOICE_SAMPLES: []},
        unique_id="speaker-recognition-main",
        version=2,
        minor_version=0,
    )


@pytest.mark.asyncio
async def test_main_entry_loads_through_real_home_assistant(hass: HomeAssistant) -> None:
    """A healthy backend allows HA to fully load the main config entry."""
    backend = BackendStub()
    await backend.start()
    try:
        entry = _main_entry(backend.url)
        entry.add_to_hass(hass)

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED
        assert entry.runtime_data is not None
        assert ("GET /health", None) in backend.requests
        assert ("POST /profiles/sync", None) in backend.requests
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_main_entry_unloads_cleanly(hass: HomeAssistant) -> None:
    """HA can unload a successfully loaded main entry without an exception."""
    backend = BackendStub()
    await backend.start()
    try:
        entry = _main_entry(backend.url)
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.NOT_LOADED
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_main_entry_reload_rebuilds_runtime(hass: HomeAssistant) -> None:
    """A real HA reload performs a fresh backend reconciliation."""
    backend = BackendStub()
    await backend.start()
    try:
        entry = _main_entry(backend.url)
        entry.add_to_hass(hass)
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        first_runtime = entry.runtime_data

        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED
        assert entry.runtime_data is not first_runtime
        assert [request[0] for request in backend.requests].count("GET /health") == 2
        assert [request[0] for request in backend.requests].count(
            "POST /profiles/sync"
        ) == 2
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_backend_unavailable_places_entry_in_setup_retry(
    hass: HomeAssistant,
) -> None:
    """Connection failure is exposed to HA as a retryable config-entry setup."""
    entry = _main_entry(f"http://127.0.0.1:{_unused_port()}")
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    assert entry.state is ConfigEntryState.SETUP_RETRY


@pytest.mark.asyncio
async def test_setup_retry_recovers_when_backend_appears(hass: HomeAssistant) -> None:
    """The same HA entry can recover through HA's retry path once backend starts."""
    port = _unused_port()
    entry = _main_entry(f"http://127.0.0.1:{port}")
    entry.add_to_hass(hass)

    assert not await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    assert entry.state is ConfigEntryState.SETUP_RETRY

    backend = BackendStub(port)
    await backend.start()
    try:
        entry.async_cancel_retry_setup()
        entry._async_setup_again(hass)  # noqa: SLF001 - exercise HA's retry callback path
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED
        assert ("GET /health", None) in backend.requests
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_invalid_health_contract_is_retryable(hass: HomeAssistant) -> None:
    """A reachable backend with an invalid health contract must not half-load."""
    backend = BackendStub()
    backend.health_payload = {"status": "healthy", "trained": "yes"}
    await backend.start()
    try:
        entry = _main_entry(backend.url)
        entry.add_to_hass(hass)

        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.SETUP_RETRY
        assert "POST /profiles/sync" not in [request[0] for request in backend.requests]
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_backend_http_failure_is_retryable(hass: HomeAssistant) -> None:
    """An HTTP error from backend health is treated as temporary unavailability."""
    backend = BackendStub()
    backend.health_status = 503
    await backend.start()
    try:
        entry = _main_entry(backend.url)
        entry.add_to_hass(hass)

        assert not await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.SETUP_RETRY
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_backend_token_is_used_for_startup_requests(hass: HomeAssistant) -> None:
    """Real HA setup sends the configured bearer token to backend endpoints."""
    backend = BackendStub()
    await backend.start()
    try:
        entry = _main_entry(backend.url, token="test-secret")
        entry.add_to_hass(hass)

        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED
        assert backend.requests == [
            ("GET /health", "Bearer test-secret"),
            ("POST /profiles/sync", "Bearer test-secret"),
        ]
    finally:
        await backend.stop()
