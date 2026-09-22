"""Real HA promotion, persistence and transactional retraining acceptance."""

from pathlib import Path
from unittest.mock import patch

import pytest
from homeassistant.components import media_source

from custom_components.speaker_recognition.const import DOMAIN
from custom_components.speaker_recognition.enrollment import (
    async_stage_pcm_sample,
    _managed_media_path,
)
from custom_components.speaker_recognition.promotions import (
    pending_promotions,
    async_setup_promotions,
    async_reconcile_promotions,
)
from custom_components.speaker_recognition.telemetry import get_decision_history
from tests_real_ha.test_options_retraining_lifecycle import StatefulBackend, _main_entry


@pytest.fixture
async def promotion_environment(hass, tmp_path, hass_ws_client):
    hass.config.media_dirs = {"local": str(tmp_path)}
    alice = await hass.auth.async_create_user("Alice")
    bob = await hass.auth.async_create_user("Bob")
    samples = []
    for user in (alice, bob):
        items = []
        for i in range(5):
            item = await async_stage_pcm_sample(
                hass, user.id, i, b"\x01\x00" * 16000, 16000
            )
            items.append({"media_content_id": item["media_content_id"]})
        samples.append({"user": user.id, "samples": items})
    hass.data[DOMAIN]["enrollment_staged"] = {}
    backend = StatefulBackend({alice.id, bob.id})
    await backend.start()
    entry = _main_entry(backend.url, samples)
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    client = await hass_ws_client(hass)
    history = get_decision_history(hass)

    def decision(feedback="missed_speaker", actual=None, audio=True):
        sequence = len(history.recent(200)) + 1
        if audio:
            hass.data[DOMAIN].setdefault("utterance_audio", {})[sequence] = (
                b"\x02\x00" * 16000,
                16000,
            )
        ident = history.record_event(
            {
                "utterance_sequence": sequence,
                "candidate_user_id": bob.id,
                "user_id": bob.id,
                "accepted": True,
                "similarity": 0.8,
                "margin": 0.1,
            }
        )
        if feedback:
            history.add_feedback(ident, feedback, actual)
        return ident

    async def send(command, **values):
        await client.send_json_auto_id({"type": f"{DOMAIN}/{command}", **values})
        return await client.receive_json()

    async def resolve(_hass, media_id, _target):
        path = _managed_media_path(hass, media_id)
        return media_source.PlayMedia(
            url="/media/sample.wav", mime_type="audio/wav", path=path
        )

    with patch(
        "custom_components.speaker_recognition.recognition.media_source.async_resolve_media",
        side_effect=resolve,
    ):
        try:
            yield alice, bob, entry, backend, history, decision, send
        finally:
            await hass.async_block_till_done()
            await backend.stop()


@pytest.mark.asyncio
async def test_promotion_persists_without_training_and_rejects_duplicates(
    hass, promotion_environment
):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    before = list(entry.options["voice_samples"])
    ident = decision(actual=alice.id)
    assert (await send("promote_decision", decision_id=ident))["success"]
    pending = pending_promotions(hass, alice.id)
    assert len(pending) == 1
    assert not pending_promotions(
        hass, bob.id
    )  # Candidate was Bob; explicit actual was Alice.
    path = _managed_media_path(hass, pending[0]["media_content_id"])
    assert path.is_file()
    assert entry.options["voice_samples"] == before
    assert not any(name == "POST /train" for name, _ in backend.requests)
    hass.data[DOMAIN].pop("promotion_store")
    hass.data[DOMAIN].pop("promoted_samples")
    await async_setup_promotions(hass)
    assert pending_promotions(hass, alice.id) == pending
    assert not (await send("promote_decision", decision_id=ident))["success"]
    assert (await send("discard_promoted_samples", user_id=alice.id))["success"]
    assert not path.exists()
    assert not (await send("promote_decision", decision_id=ident))["success"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["no_feedback", "unknown", "expired", "unenrolled", "inactive"]
)
async def test_ineligible_promotion(hass, promotion_environment, case):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    actual = alice.id
    if case == "unenrolled":
        actual = (await hass.auth.async_create_user("Not enrolled")).id
    if case == "inactive":
        await hass.auth.async_deactivate_user(alice)
    ident = decision(
        feedback=None if case == "no_feedback" else "wrong_speaker",
        actual=None if case == "unknown" else actual,
        audio=case != "expired",
    )
    response = await send("promote_decision", decision_id=ident)
    assert not response["success"]
    assert pending_promotions(hass) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_promoted_retraining_transaction(hass, promotion_environment, fail):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    original = list(entry.options["voice_samples"])
    original_paths = [
        _managed_media_path(hass, item["media_content_id"])
        for row in original
        for item in row["samples"]
    ]
    ident = decision(actual=alice.id)
    assert (await send("promote_decision", decision_id=ident))["success"]
    promoted_path = _managed_media_path(
        hass, pending_promotions(hass)[0]["media_content_id"]
    )
    backend.fail_training = fail
    response = await send("commit_enrollment", user_id=alice.id)
    await hass.async_block_till_done()
    assert response["success"] is not fail
    assert all(path.exists() for path in original_paths)
    assert promoted_path.exists()
    assert entry.options["voice_samples"][1] == original[1]
    assert backend.enrolled_users == {alice.id, bob.id}
    calls = [payload for name, payload in backend.requests if name == "POST /train"]
    assert len(calls) == 1
    assert {item["user"] for item in calls[0]["voice_samples"]} == {alice.id}
    assert len(calls[0]["voice_samples"]) == 6
    if fail:
        assert entry.options["voice_samples"] == original
        assert len(pending_promotions(hass)) == 1
    else:
        assert len(entry.options["voice_samples"][0]["samples"]) == 6
        assert pending_promotions(hass) == []
        assert not (await send("promote_decision", decision_id=ident))["success"]


@pytest.mark.asyncio
async def test_removed_user_pending_media_cleanup(hass, promotion_environment):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    assert (await send("promote_decision", decision_id=decision(actual=alice.id)))[
        "success"
    ]
    path = _managed_media_path(hass, pending_promotions(hass)[0]["media_content_id"])
    await async_reconcile_promotions(hass, [entry.options["voice_samples"][1]])
    assert not path.exists()
    assert pending_promotions(hass) == []

@pytest.mark.asyncio
async def test_promotion_bound_and_review_queue_unchanged(hass, promotion_environment):
    alice, bob, entry, backend, history, decision, send = promotion_environment
    for i in range(7):
        response = await send("promote_decision", decision_id=decision(actual=alice.id))
        assert response["success"] is (i < 6)
    assert len(pending_promotions(hass)) == 6
    for _ in range(10):
        decision(actual=alice.id)
    assert len(history.review_audio_ids()) == 10
    assert len(pending_promotions(hass)) == 6


@pytest.mark.asyncio
@pytest.mark.parametrize("command,values", [
    ("promote_decision", {"decision_id": "anything"}),
    ("discard_promoted_samples", {"user_id": "anything"}),
])
async def test_promotion_commands_require_admin(hass, promotion_environment, hass_ws_client, hass_read_only_access_token, command, values):
    client = await hass_ws_client(hass, hass_read_only_access_token)
    await client.send_json_auto_id({"type": f"{DOMAIN}/{command}", **values})
    assert (await client.receive_json())["error"]["code"] == "unauthorized"

@pytest.mark.asyncio
async def test_promoted_and_phrase_samples_share_advisory_preview(hass, promotion_environment):
    alice, bob, entry, backend, history, decision, send = promotion_environment

    async def quality(path, payload, **kwargs):
        assert path == "/enrollment/quality"
        return {"samples": [{"sample_index": i + 1, "assessment": "insufficient_evidence"}
                            for i, _ in enumerate(payload["voice_samples"])]}

    with patch.object(entry.runtime_data, "_async_post", side_effect=quality):
        assert (await send("promote_decision", decision_id=decision(actual=alice.id)))["success"]
        await hass.async_block_till_done()
        assert set(hass.data[DOMAIN]["enrollment_quality"][alice.id]["samples"]) == {6}
        await async_stage_pcm_sample(hass, alice.id, 0, b"\x03\x00" * 16000, 16000)
        await hass.async_block_till_done()
        assert set(hass.data[DOMAIN]["enrollment_quality"][alice.id]["samples"]) == {0, 6}
        assert (await send("discard_promoted_samples", user_id=alice.id))["success"]
        await hass.async_block_till_done()
        assert set(hass.data[DOMAIN]["enrollment_quality"][alice.id]["samples"]) == {0}
