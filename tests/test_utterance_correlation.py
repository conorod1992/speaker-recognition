"""Regression tests for per-Assist-turn speaker correlation."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path


def _load_correlation_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "correlation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_integration_correlation", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(module, user_id: str, sequence: int):
    return module.CorrelatedRecognition(
        user_id=user_id,
        candidate_user_id=user_id,
        confidence=0.9,
        similarity=0.9,
        margin=0.3,
        accepted=True,
        all_scores={user_id: 0.9},
        stt_entity_id="stt.speaker_recognition",
        utterance_sequence=sequence,
    )


def test_correlation_is_consumed_once() -> None:
    """A conversation turn cannot accidentally reuse a prior STT decision."""
    module = _load_correlation_module()
    decision = _decision(module, "alice", 1)

    module.set_correlated_recognition(decision)

    assert module.take_correlated_recognition() == decision
    assert module.take_correlated_recognition() is None


def test_clear_removes_inherited_or_stale_decision() -> None:
    """Starting a new STT turn clears any task-local decision left behind."""
    module = _load_correlation_module()
    module.set_correlated_recognition(_decision(module, "alice", 1))

    module.clear_correlated_recognition()

    assert module.take_correlated_recognition() is None


def test_concurrent_assist_tasks_keep_their_own_speaker() -> None:
    """Overlapping pipelines cannot steal one another's recognition result."""
    module = _load_correlation_module()

    async def run() -> tuple[str, str]:
        first_ready = asyncio.Event()
        second_ready = asyncio.Event()

        async def first() -> str:
            module.clear_correlated_recognition()
            module.set_correlated_recognition(_decision(module, "alice", 1))
            first_ready.set()
            await second_ready.wait()
            result = module.take_correlated_recognition()
            assert result is not None
            return result.user_id or ""

        async def second() -> str:
            await first_ready.wait()
            module.clear_correlated_recognition()
            module.set_correlated_recognition(_decision(module, "bob", 2))
            second_ready.set()
            await asyncio.sleep(0)
            result = module.take_correlated_recognition()
            assert result is not None
            return result.user_id or ""

        first_user, second_user = await asyncio.gather(first(), second())
        return first_user, second_user

    assert asyncio.run(run()) == ("alice", "bob")


def test_integration_no_longer_uses_global_last_result_cache() -> None:
    """The STT/conversation hand-off must not regress to a shared global cache."""
    root = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
    stt_source = (root / "stt.py").read_text(encoding="utf-8")
    conversation_source = (root / "conversation.py").read_text(encoding="utf-8")

    assert 'domain_data["last_result"]' not in stt_source
    assert '.get("last_result")' not in conversation_source
    assert "set_correlated_recognition" in stt_source
    assert "take_correlated_recognition" in conversation_source
