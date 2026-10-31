"""Runtime health validation tests for the Home Assistant integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / "speaker_recognition"


def _load_runtime(monkeypatch: pytest.MonkeyPatch):
    package_name = "test_speaker_recognition_runtime"
    package = ModuleType(package_name)
    package.__path__ = [str(INTEGRATION)]

    live_evaluation = ModuleType(f"{package_name}.live_evaluation")

    class BaseRecognition:
        def __init__(self, response: dict) -> None:
            self.response = response

        async def _async_get(self, path: str) -> dict:
            assert path == "/health"
            return dict(self.response)

    live_evaluation.LiveEvaluationSpeakerRecognition = BaseRecognition
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(
        sys.modules, f"{package_name}.live_evaluation", live_evaluation
    )

    spec = importlib.util.spec_from_file_location(
        f"{package_name}.runtime", INTEGRATION / "runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_runtime_accepts_ready_authoritative_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runtime(monkeypatch)
    recognition = module.LiveEvaluationSpeakerRecognition(
        {"status": "healthy", "encoder_ready": True}
    )

    response = await recognition._async_get("/health")

    assert response["encoder_ready"] is True


@pytest.mark.asyncio
async def test_runtime_rejects_degraded_authoritative_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runtime(monkeypatch)
    recognition = module.LiveEvaluationSpeakerRecognition(
        {
            "status": "degraded",
            "encoder_ready": False,
            "warmup_error": "model unavailable",
        }
    )

    with pytest.raises(ValueError, match="model unavailable"):
        await recognition._async_get("/health")


@pytest.mark.asyncio
async def test_runtime_rejects_health_without_readiness_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runtime(monkeypatch)
    recognition = module.LiveEvaluationSpeakerRecognition({"status": "healthy"})

    with pytest.raises(ValueError, match="omitted encoder readiness"):
        await recognition._async_get("/health")


def test_main_entry_uses_readiness_validating_runtime() -> None:
    source = (INTEGRATION / "__init__.py").read_text(encoding="utf-8")
    assert "from .runtime import LiveEvaluationSpeakerRecognition" in source


def test_health_endpoint_does_not_block_behind_training_lock() -> None:
    """Readiness probes must be responsive while a training job owns the lock."""
    source = (ROOT / "speaker_recognition" / "api.py").read_text(encoding="utf-8")
    health = source.split("def health_check()", 1)[1].split(
        '@app.get("/profiles/diagnostics"', 1
    )[0]
    assert "_RECOGNIZER_LOCK.acquire(blocking=False)" in health
    assert "_LAST_HEALTH_PROFILES" in health
    assert "with _RECOGNIZER_LOCK:" not in health


def test_health_responds_while_authoritative_training_lock_is_held() -> None:
    """Exercise the health endpoint under actual model-lock contention."""
    import importlib
    from time import perf_counter

    api = importlib.import_module("speaker_recognition.api")
    assert api._RECOGNIZER_LOCK.acquire(blocking=False)
    try:
        start = perf_counter()
        response = api.health_check()
        elapsed = perf_counter() - start
    finally:
        api._RECOGNIZER_LOCK.release()
    assert elapsed < 0.5
    assert response.status in ("healthy", "degraded")
    assert isinstance(response.enrolled_users, list)
