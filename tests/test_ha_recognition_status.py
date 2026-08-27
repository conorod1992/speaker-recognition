"""Status-driven availability tests for the Home Assistant client wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _load_recognition_module(monkeypatch: pytest.MonkeyPatch):
    integration_path = (
        Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
    )
    package_name = "test_speaker_recognition_integration"
    package = ModuleType(package_name)
    package.__path__ = [str(integration_path)]

    aiohttp = ModuleType("aiohttp")

    class ClientError(Exception):
        pass

    class ClientTimeout:
        def __init__(self, *, total: int) -> None:
            self.total = total

    aiohttp.ClientError = ClientError
    aiohttp.ClientTimeout = ClientTimeout
    homeassistant = ModuleType("homeassistant")
    components = ModuleType("homeassistant.components")
    media_source = ModuleType("homeassistant.components.media_source")
    helpers = ModuleType("homeassistant.helpers")
    aiohttp_client = ModuleType("homeassistant.helpers.aiohttp_client")
    aiohttp_client.async_get_clientsession = lambda hass: None
    components.media_source = media_source

    for name, module in {
        package_name: package,
        "aiohttp": aiohttp,
        "homeassistant": homeassistant,
        "homeassistant.components": components,
        "homeassistant.components.media_source": media_source,
        "homeassistant.helpers": helpers,
        "homeassistant.helpers.aiohttp_client": aiohttp_client,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = integration_path / "recognition.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.recognition", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeHass:
    async def async_add_executor_job(self, target, *args):
        return target(*args)


@pytest.mark.asyncio
async def test_persisted_backend_status_enables_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh HA wrapper recognizes using profiles loaded by the add-on."""
    module = _load_recognition_module(monkeypatch)
    recognition = module.SpeakerRecognition(_FakeHass(), [])

    async def async_get(path: str):
        assert path == "/health"
        return {"status": "healthy", "trained": True, "enrolled_users": ["alice"]}

    async def async_post(path: str, payload):
        del payload
        assert path == "/recognize"
        return {"user_id": "alice", "confidence": 0.9, "all_scores": {"alice": 0.9}}

    recognition._async_get = async_get
    recognition._async_post = async_post

    assert await recognition.async_refresh_status()
    result = await recognition.async_recognize(b"\x01\x00")
    assert result is not None
    assert result.user_id == "alice"


@pytest.mark.asyncio
async def test_backend_status_failure_is_explicitly_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup can distinguish an unavailable backend from an untrained backend."""
    module = _load_recognition_module(monkeypatch)
    recognition = module.SpeakerRecognition(_FakeHass(), [])

    async def async_get(path: str):
        del path
        raise OSError("backend starting")

    recognition._async_get = async_get

    with pytest.raises(module.RecognitionBackendUnavailable):
        await recognition.async_refresh_status()


@pytest.mark.asyncio
async def test_failed_training_keeps_loaded_profiles_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HA does not globally disable recognition when one enrollment request fails."""
    module = _load_recognition_module(monkeypatch)
    recognition = module.SpeakerRecognition(
        _FakeHass(),
        [
            {
                "user": "alice",
                "samples": [{"media_content_id": "media-source://alice.wav"}],
            }
        ],
    )

    async def async_get(path: str):
        del path
        return {"status": "healthy", "trained": True, "enrolled_users": ["bob"]}

    async def async_read_media(media_id: str):
        del media_id
        return b"wav"

    async def async_post(path: str, payload):
        del payload
        if path == "/train":
            raise OSError("training failed")
        return {"user_id": "bob", "confidence": 0.8, "all_scores": {"bob": 0.8}}

    monkeypatch.setattr(module, "decode_wav", lambda audio: (b"\x01\x00", 16000))
    recognition._async_get = async_get
    recognition._async_read_media = async_read_media
    recognition._async_post = async_post

    assert await recognition.async_refresh_status()
    assert not await recognition.async_train({"alice"})
    result = await recognition.async_recognize(b"\x01\x00")
    assert result is not None
    assert result.user_id == "bob"
