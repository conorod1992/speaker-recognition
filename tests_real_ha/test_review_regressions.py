"""Behavioral regressions for persisted turns and serialized runtime updates."""

from __future__ import annotations

import asyncio
from threading import Event, get_ident
from unittest.mock import AsyncMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry
from homeassistant.setup import async_setup_component

from custom_components.speaker_recognition import async_update_main_listener
from custom_components.speaker_recognition.const import DOMAIN
from custom_components.speaker_recognition.live_evaluation import LiveModelEvaluation
from custom_components.speaker_recognition.telemetry import DecisionHistory


@pytest.mark.asyncio
async def test_restart_does_not_reuse_labelled_turn_or_audio(hass):
    history = DecisionHistory(hass)
    hass.data[DOMAIN] = {"utterance_audio": {1: (b"old clip", 16000)}}
    old_id = history.record_event({
        "utterance_sequence": 1, "entity_id": "stt.proxy",
        "candidate_user_id": "alice", "user_id": "alice", "accepted": True,
    })
    history.add_feedback(old_id, "correct", None)
    await history._store.async_save({"records": history._records})
    await history._audio_store.async_save({"clips": history._review_audio})
    hass.data[DOMAIN] = {}
    restarted = DecisionHistory(hass)
    await restarted.async_load()
    sequence = hass.data[DOMAIN]["utterance_sequence"] + 1
    hass.data[DOMAIN]["utterance_audio"] = {sequence: (b"new clip", 16000)}
    new_id = restarted.record_event({
        "utterance_sequence": sequence, "entity_id": "stt.proxy",
        "candidate_user_id": "bob", "user_id": "bob", "accepted": True,
    })
    assert new_id != old_id
    assert len(restarted.recent()) == 2
    assert restarted.recent()[0]["feedback"] is None
    assert restarted.labelled()[0]["candidate_user_id"] == "alice"
    assert restarted.review_audio_for_decision(old_id) != restarted.review_audio_for_decision(new_id)


@pytest.mark.asyncio
async def test_evaluation_polls_coalesce_off_loop_and_invalidate_on_changes(hass):
    evaluation = LiveModelEvaluation(hass)
    loop_thread = get_ident()
    counts = []

    def analyze(records):
        assert get_ident() != loop_thread
        counts.append(len(records))
        return {"trial_count": len(records), "nested": {"value": "original"}}

    with patch("custom_components.speaker_recognition.live_evaluation.analyze_live_evaluation", analyze):
        first, second = await asyncio.gather(evaluation.async_analysis(), evaluation.async_analysis())
        first["nested"]["value"] = "changed"
        assert second["nested"]["value"] == "original"
        assert counts == [0]
        evaluation.pending = {"authoritative": {}, "shadow": {}}
        evaluation.label_pending("alice")
        assert (await evaluation.async_analysis())["trial_count"] == 1
        evaluation.clear()
        assert (await evaluation.async_analysis())["trial_count"] == 0
        assert counts == [0, 1, 0]


@pytest.mark.asyncio
async def test_evaluation_discards_analysis_of_a_cleared_snapshot(hass):
    evaluation = LiveModelEvaluation(hass)
    evaluation._records = [{"actual_user_id": "alice"}]
    started, release = asyncio.Event(), Event()
    calls = []

    def analyze(records):
        calls.append(len(records))
        if len(calls) == 1:
            hass.loop.call_soon_threadsafe(started.set)
            assert release.wait(3)
        return {"trial_count": len(records)}

    with patch("custom_components.speaker_recognition.live_evaluation.analyze_live_evaluation", analyze):
        task = asyncio.create_task(evaluation.async_analysis())
        try:
            await asyncio.wait_for(started.wait(), 1)
            evaluation.clear()
        finally:
            release.set()
        assert (await task)["trial_count"] == 0
    assert calls == [1, 0]


@pytest.mark.asyncio
async def test_async_evaluation_status_websocket_returns_a_result(hass, hass_ws_client):
    assert await async_setup_component(hass, DOMAIN, {})
    client = await hass_ws_client(hass)
    await client.send_json_auto_id({"type": f"{DOMAIN}/evaluation_status"})
    result = await client.receive_json()
    assert result["success"] is True
    assert result["result"]["trial_count"] == 0


@pytest.mark.asyncio
async def test_second_users_sidebar_commit_is_rejected_while_training(hass, hass_ws_client):
    assert await async_setup_component(hass, DOMAIN, {})
    user = await hass.auth.async_create_user("Bob")
    entry = MockConfigEntry(domain=DOMAIN, data={"entry_type": "main"}, options={"voice_samples": []})
    entry.add_to_hass(hass)
    hass.data[DOMAIN]["enrollment_staged"] = {user.id: {index: {"media_content_id": f"sample-{index}"} for index in range(5)}}
    completion = hass.loop.create_future()
    hass.data[DOMAIN]["enrollment_commit_waiters"] = {"alice": completion}
    client = await hass_ws_client(hass)
    try:
        await client.send_json_auto_id({"type": f"{DOMAIN}/commit_enrollment", "user_id": user.id})
        result = await client.receive_json()
        assert result["success"] is False
        assert result["error"]["code"] == "commit_in_progress"
        assert entry.options["voice_samples"] == []
    finally:
        completion.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_user", ["alice", "bob"])
async def test_failed_enrollment_cannot_erase_queued_update(hass, newer_user):
    old = [{"user": user, "samples": [f"old-{user}"]} for user in ("alice", "bob")]
    first = [{"user": "alice", "samples": ["first-alice"]}, old[1]]
    latest = [
        {"user": row["user"], "samples": [f"new-{newer_user}"]}
        if row["user"] == newer_user else row for row in first
    ]
    started = asyncio.Event()
    finish = asyncio.Event()
    calls = []

    class Runtime:
        voice_samples = old

        def update_voice_samples(self, samples):
            self.voice_samples = samples

        async def async_train(self, users):
            calls.append(users)
            if len(calls) == 1:
                started.set()
                await finish.wait()
                return False
            return True

        async def async_sync_profiles(self):
            return True

    entry = MockConfigEntry(domain=DOMAIN, data={"entry_type": "main"}, options={"voice_samples": first})
    entry.add_to_hass(hass)
    entry.runtime_data = Runtime()
    with (
        patch.object(hass.config_entries, "async_reload", new=AsyncMock(return_value=True)),
        patch("custom_components.speaker_recognition.async_cleanup_managed_samples", new=AsyncMock()),
        patch("custom_components.speaker_recognition.async_reconcile_promotions", new=AsyncMock()),
    ):
        first_task = asyncio.create_task(async_update_main_listener(hass, entry))
        await asyncio.wait_for(started.wait(), 1)
        hass.config_entries.async_update_entry(entry, options={"voice_samples": latest, "acceptance_thresholds": {"min_similarity": .9, "min_margin": .3}})
        second_task = asyncio.create_task(async_update_main_listener(hass, entry))
        await asyncio.sleep(0)
        assert len(calls) == 1
        finish.set()
        await asyncio.gather(first_task, second_task)
    expected = {row["user"]: row for row in old}
    expected[newer_user] = {"user": newer_user, "samples": [f"new-{newer_user}"]}
    assert {row["user"]: row for row in entry.options["voice_samples"]} == expected
    assert {row["user"]: row for row in entry.runtime_data.voice_samples} == expected
    assert calls == [{"alice"}, {newer_user}]
    assert entry.options["acceptance_thresholds"]["min_similarity"] == .9
