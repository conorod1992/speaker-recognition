"""Real Home Assistant acceptance tests for options and profile lifecycle."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from unittest.mock import patch
import wave

from aiohttp import web
import pytest
from homeassistant import config_entries
from homeassistant.components import media_source
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.speaker_recognition.const import (
    CONF_BACKEND_TOKEN,
    CONF_BACKEND_URL,
    CONF_ENTRY_TYPE,
    CONF_FINISH_ENROLLMENT,
    CONF_SAMPLE,
    CONF_SAMPLES,
    CONF_USER,
    CONF_VOICE_SAMPLES,
    DOMAIN,
    ENTRY_TYPE_MAIN,
)


class StatefulBackend:
    """Small stateful backend for exercising real HA profile updates."""

    def __init__(self, enrolled_users: set[str] | None = None) -> None:
        self.port = 0
        self.enrolled_users = set(enrolled_users or set())
        self.fail_training = False
        self.requests: list[tuple[str, dict[str, Any] | None]] = []
        self._runner: web.AppRunner | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/train", self._train)
        app.router.add_post("/profiles/sync", self._sync)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets if site._server is not None else []  # noqa: SLF001
        assert sockets
        self.port = int(sockets[0].getsockname()[1])

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _health(self, request: web.Request) -> web.Response:
        self.requests.append(("GET /health", None))
        return web.json_response(
            {
                "status": "healthy",
                "trained": bool(self.enrolled_users),
                "enrolled_users": sorted(self.enrolled_users),
                "encoder_ready": True,
                "warmup_error": None,
            }
        )

    async def _train(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.requests.append(("POST /train", payload))
        if self.fail_training:
            return web.json_response({"error": "training failed"}, status=500)

        models = payload.get("voice_samples", [])
        counts = Counter(
            item.get("user")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("user"), str)
        )
        trained_users = sorted(counts)
        self.enrolled_users.update(trained_users)
        return web.json_response(
            {
                "trained_users": trained_users,
                "accepted_samples": dict(counts),
            }
        )

    async def _sync(self, request: web.Request) -> web.Response:
        payload = await request.json()
        self.requests.append(("POST /profiles/sync", payload))
        desired = set(payload.get("desired_users", []))
        removed = sorted(self.enrolled_users - desired)
        self.enrolled_users.intersection_update(desired)
        return web.json_response(
            {
                "enrolled_users": sorted(self.enrolled_users),
                "removed_users": removed,
            }
        )


def _write_wav(path: Path, *, frames: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x01\x00" * frames)


def _media_item(name: str) -> dict[str, str]:
    return {
        "media_content_id": f"media-source://media_source/local/{name}",
        "media_content_type": "audio/wav",
    }


def _enrollment(user_id: str, prefix: str) -> dict[str, Any]:
    return {
        CONF_USER: user_id,
        CONF_SAMPLES: [_media_item(f"{prefix}-{index}.wav") for index in range(5)],
        "sample_metadata": [{"phrase": f"phrase-{index}"} for index in range(5)],
    }


def _main_entry(
    backend_url: str,
    voice_samples: list[dict[str, Any]],
    *,
    token: str = "",
) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        title="Speaker Recognition",
        data={
            CONF_ENTRY_TYPE: ENTRY_TYPE_MAIN,
            CONF_BACKEND_URL: backend_url,
            CONF_BACKEND_TOKEN: token,
        },
        options={CONF_VOICE_SAMPLES: voice_samples},
        unique_id=ENTRY_TYPE_MAIN,
        version=2,
        minor_version=0,
    )


async def _start_options_flow(hass: HomeAssistant, entry: MockConfigEntry) -> dict:
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    return result


async def _open_options_enrollment(hass: HomeAssistant, entry: MockConfigEntry) -> dict:
    result = await _start_options_flow(hass, entry)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "enrollment_user"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "enrollment_user"
    return result


async def _complete_five_sample_enrollment(
    hass: HomeAssistant,
    result: dict,
    user_id: str,
    wav_path: Path,
    prefix: str,
) -> dict:
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {CONF_USER: user_id}
    )
    assert result["step_id"] == "enrollment_sample"

    with patch(
        "custom_components.speaker_recognition.config_flow.media_source.async_resolve_media",
        return_value=media_source.PlayMedia(
            url=f"/media/{wav_path.name}", mime_type="audio/wav", path=wav_path
        ),
    ):
        for index in range(5):
            user_input: dict[str, Any] = {
                CONF_SAMPLE: _media_item(f"{prefix}-{index}.wav")
            }
            if index == 4:
                user_input[CONF_FINISH_ENROLLMENT] = True
            result = await hass.config_entries.options.async_configure(
                result["flow_id"], user_input
            )

    assert result["step_id"] == "enrollment_review"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"enrollment_action": "finish"}
    )
    assert result["step_id"] == "enrollment_complete"
    return await hass.config_entries.options.async_configure(result["flow_id"], {})


@pytest.mark.asyncio
async def test_main_options_load_current_values_and_preserve_enrollment(
    hass: HomeAssistant,
) -> None:
    """Main options use effective values and do not discard existing speakers."""
    existing = _enrollment("alice", "old")
    entry = _main_entry("http://old.invalid", [existing], token="old-token")
    entry.add_to_hass(hass)

    result = await _start_options_flow(hass, entry)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "main_options"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "main_options"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {CONF_BACKEND_URL: "http://new.invalid", CONF_BACKEND_TOKEN: "new-token"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_BACKEND_URL] == "http://new.invalid"
    assert entry.options[CONF_BACKEND_TOKEN] == "new-token"
    assert entry.options[CONF_VOICE_SAMPLES] == [existing]


@pytest.mark.asyncio
async def test_options_enrollment_adds_second_user_without_changing_first(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Guided options enrollment adds a second HA user alongside the first."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    alice_enrollment = _enrollment(alice.id, "alice-old")
    entry = _main_entry("http://backend.invalid", [alice_enrollment])
    entry.add_to_hass(hass)
    wav_path = tmp_path / "bob.wav"
    _write_wav(wav_path)

    result = await _complete_five_sample_enrollment(
        hass, await _open_options_enrollment(hass, entry), bob.id, wav_path, "bob"
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    by_user = {item[CONF_USER]: item for item in entry.options[CONF_VOICE_SAMPLES]}
    assert by_user[alice.id] == alice_enrollment
    assert by_user[bob.id][CONF_SAMPLES] == [
        _media_item(f"bob-{index}.wav") for index in range(5)
    ]


@pytest.mark.asyncio
async def test_options_retraining_replaces_only_selected_user(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Retraining Alice leaves Bob's configured sample set untouched."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    old_alice = _enrollment(alice.id, "alice-old")
    bob_enrollment = _enrollment(bob.id, "bob-old")
    entry = _main_entry("http://backend.invalid", [old_alice, bob_enrollment])
    entry.add_to_hass(hass)
    wav_path = tmp_path / "alice-new.wav"
    _write_wav(wav_path)

    result = await _complete_five_sample_enrollment(
        hass, await _open_options_enrollment(hass, entry), alice.id, wav_path, "alice-new"
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    by_user = {item[CONF_USER]: item for item in entry.options[CONF_VOICE_SAMPLES]}
    assert by_user[bob.id] == bob_enrollment
    assert by_user[alice.id] != old_alice
    assert by_user[alice.id][CONF_SAMPLES] == [
        _media_item(f"alice-new-{index}.wav") for index in range(5)
    ]


@pytest.mark.asyncio
async def test_successful_retraining_trains_changed_user_and_reloads_runtime(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """A loaded main entry trains changed samples, syncs, and reloads in real HA."""
    user = await hass.auth.async_create_user("Alice")
    old = _enrollment(user.id, "old")
    new = _enrollment(user.id, "new")
    backend = StatefulBackend({user.id})
    await backend.start()
    wav_path = tmp_path / "new.wav"
    _write_wav(wav_path)
    entry = _main_entry(backend.url, [old])
    entry.add_to_hass(hass)

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        first_runtime = entry.runtime_data
        backend.requests.clear()

        with patch(
            "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="/media/new.wav", mime_type="audio/wav", path=wav_path
            ),
        ):
            hass.config_entries.async_update_entry(
                entry, options={CONF_VOICE_SAMPLES: [new]}
            )
            await hass.async_block_till_done(wait_background_tasks=True)

        assert entry.state is ConfigEntryState.LOADED
        assert entry.runtime_data is not first_runtime
        train_payloads = [payload for path, payload in backend.requests if path == "POST /train"]
        assert len(train_payloads) == 1
        assert {item["user"] for item in train_payloads[0]["voice_samples"]} == {user.id}
        assert len(train_payloads[0]["voice_samples"]) == 5
        assert entry.options[CONF_VOICE_SAMPLES] == [new]
        assert backend.enrolled_users == {user.id}
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_failed_retraining_rolls_back_options_and_keeps_existing_profile(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Training failure restores old options without sacrificing the usable profile."""
    user = await hass.auth.async_create_user("Alice")
    old = _enrollment(user.id, "old")
    new = _enrollment(user.id, "new")
    backend = StatefulBackend({user.id})
    backend.fail_training = True
    await backend.start()
    wav_path = tmp_path / "new.wav"
    _write_wav(wav_path)
    entry = _main_entry(backend.url, [old])
    entry.add_to_hass(hass)

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        with patch(
            "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="/media/new.wav", mime_type="audio/wav", path=wav_path
            ),
        ):
            hass.config_entries.async_update_entry(
                entry, options={CONF_VOICE_SAMPLES: [new]}
            )
            await hass.async_block_till_done(wait_background_tasks=True)

        assert entry.options[CONF_VOICE_SAMPLES] == [old]
        assert entry.runtime_data.voice_samples == [old]
        assert backend.enrolled_users == {user.id}
        assert any(path == "POST /train" for path, _ in backend.requests)
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_removing_user_syncs_backend_and_reloads_remaining_configuration(
    hass: HomeAssistant,
) -> None:
    """Removing one configured user deletes only that stale backend profile."""
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    alice_samples = _enrollment(alice.id, "alice")
    bob_samples = _enrollment(bob.id, "bob")
    backend = StatefulBackend({alice.id, bob.id})
    await backend.start()
    entry = _main_entry(backend.url, [alice_samples, bob_samples])
    entry.add_to_hass(hass)

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        backend.requests.clear()

        hass.config_entries.async_update_entry(
            entry, options={CONF_VOICE_SAMPLES: [alice_samples]}
        )
        await hass.async_block_till_done(wait_background_tasks=True)

        assert entry.state is ConfigEntryState.LOADED
        assert entry.options[CONF_VOICE_SAMPLES] == [alice_samples]
        assert backend.enrolled_users == {alice.id}
        assert not any(path == "POST /train" for path, _ in backend.requests)
        sync_payloads = [
            payload for path, payload in backend.requests if path == "POST /profiles/sync"
        ]
        assert any(payload == {"desired_users": [alice.id]} for payload in sync_payloads)
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_successful_retraining_deletes_superseded_managed_media(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Old integration-owned enrollment audio is removed after replacement succeeds."""
    user = await hass.auth.async_create_user("Alice")
    media_root = tmp_path / "media"
    hass.config.media_dirs = {"local": str(media_root)}
    old_path = media_root / "speaker_recognition_enrollment" / user.id / "old.wav"
    _write_wav(old_path)
    old = {
        CONF_USER: user.id,
        CONF_SAMPLES: [
            {
                "media_content_id": (
                    f"media-source://media_source/local/"
                    f"speaker_recognition_enrollment/{user.id}/old.wav"
                )
            }
        ],
    }
    new = _enrollment(user.id, "new")
    backend = StatefulBackend({user.id})
    await backend.start()
    new_path = tmp_path / "new.wav"
    _write_wav(new_path)
    entry = _main_entry(backend.url, [old])
    entry.add_to_hass(hass)

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        assert old_path.exists()

        with patch(
            "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="/media/new.wav", mime_type="audio/wav", path=new_path
            ),
        ):
            hass.config_entries.async_update_entry(
                entry, options={CONF_VOICE_SAMPLES: [new]}
            )
            await hass.async_block_till_done(wait_background_tasks=True)

        assert not old_path.exists()
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_retraining_never_deletes_unrelated_media(
    hass: HomeAssistant, tmp_path: Path
) -> None:
    """Cleanup is confined to integration-owned enrollment paths."""
    user = await hass.auth.async_create_user("Alice")
    media_root = tmp_path / "media"
    hass.config.media_dirs = {"local": str(media_root)}
    unrelated_path = media_root / "family" / "keep.wav"
    _write_wav(unrelated_path)
    old = {
        CONF_USER: user.id,
        CONF_SAMPLES: [
            {
                "media_content_id": "media-source://media_source/local/family/keep.wav"
            }
        ],
    }
    new = _enrollment(user.id, "new")
    backend = StatefulBackend({user.id})
    await backend.start()
    new_path = tmp_path / "new.wav"
    _write_wav(new_path)
    entry = _main_entry(backend.url, [old])
    entry.add_to_hass(hass)

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        with patch(
            "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="/media/new.wav", mime_type="audio/wav", path=new_path
            ),
        ):
            hass.config_entries.async_update_entry(
                entry, options={CONF_VOICE_SAMPLES: [new]}
            )
            await hass.async_block_till_done(wait_background_tasks=True)

        assert unrelated_path.exists()
    finally:
        await backend.stop()
