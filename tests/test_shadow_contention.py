"""Shadow contention must never consume the authoritative API's worker pool."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
async def test_shadow_burst_fails_fast_and_health_remains_responsive(monkeypatch):
    from speaker_recognition import api

    shadow = SimpleNamespace(
        engine_id="ecapa_tdnn", engine_name="Test", is_trained=True, enrolled_users=[],
    )
    monkeypatch.setattr(api, "shadow_service", SimpleNamespace(enabled=True, recognizer=shadow, last_error=None))
    transport = httpx.ASGITransport(app=api.app)
    payloads = [
        ("/shadow/recognize", {"audio": {"audio_data": "AA==", "sample_rate": 16000}}),
        ("/shadow/train", {"voice_samples": [{"user": "alice", "audio": {"audio_data": "AA==", "sample_rate": 16000}}]}),
        ("/shadow/profiles/sync", {"desired_users": []}),
    ]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        api._SHADOW_LOCK.acquire()
        tasks = [asyncio.create_task(client.post(path, json=body)) for path, body in payloads for _ in range(20)]
        try:
            responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
            assert {response.status_code for response in responses} == {503}
            assert (await asyncio.wait_for(client.get("/health"), 1)).status_code == 200
            # Authoritative work is still dispatched while the shadow lock is held.
            response = await asyncio.wait_for(client.post("/recognize", json=payloads[0][1]), 1)
            assert response.status_code != 503
        finally:
            api._SHADOW_LOCK.release()
            await asyncio.gather(*tasks, return_exceptions=True)
