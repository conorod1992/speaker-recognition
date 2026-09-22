"""Real HA capture preserves recordings while advisory backend analysis runs."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.speaker_recognition.const import DOMAIN
from custom_components.speaker_recognition.enrollment import (
    async_stage_pcm_sample,
    staged_samples,
    _managed_media_path,
)
from tests_real_ha.test_main_entry_lifecycle import BackendStub, _main_entry


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_capture_quality_and_backend_failure(
    hass, tmp_path, hass_ws_client, fail
):
    hass.config.media_dirs = {"local": str(tmp_path)}
    user = await hass.auth.async_create_user("Preview User")
    backend = BackendStub()
    await backend.start()
    entry = _main_entry(backend.url)
    entry.add_to_hass(hass)
    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        result = {
            "consistency": None,
            "samples": [
                {
                    "sample_index": 1,
                    "assessment": "insufficient_evidence",
                    "profile_similarity": None,
                }
            ],
        }
        mocked = (
            AsyncMock(side_effect=RuntimeError("backend unavailable"))
            if fail
            else AsyncMock(return_value=result)
        )
        with patch.object(entry.runtime_data, "_async_post", mocked):
            quality = await async_stage_pcm_sample(
                hass, user.id, 2, b"\x01\x00" * 16000, 16000
            )
            await hass.async_block_till_done()
        assert _managed_media_path(hass, quality["media_content_id"]).is_file()
        assert 2 in staged_samples(hass, user.id)
        client = await hass_ws_client(hass)
        await client.send_json_auto_id({"type": f"{DOMAIN}/status"})
        preview = (await client.receive_json())["result"]["enrollment_quality"][user.id]
        assert preview["state"] == ("unavailable" if fail else "ready")
        if not fail:
            assert preview["samples"]["2"]["assessment"] == "insufficient_evidence"
        mocked.assert_awaited_once()
        assert mocked.await_args.args[0] == "/enrollment/quality"
        assert entry.options.get("voice_samples", []) == []
    finally:
        await backend.stop()


@pytest.mark.asyncio
async def test_preview_coalesces_retake_and_cancels_on_unload(hass, tmp_path):
    hass.config.media_dirs = {"local": str(tmp_path)}
    user = await hass.auth.async_create_user("Retake User")
    backend = BackendStub()
    await backend.start()
    entry = _main_entry(backend.url)
    entry.add_to_hass(hass)
    started, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def analyze(*args, **kwargs):
        nonlocal count
        count += 1
        started.set()
        await release.wait()
        return {
            "samples": [
                {
                    "sample_index": 1,
                    "assessment": "good" if count == 2 else "inconsistent",
                }
            ]
        }

    try:
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        with patch.object(entry.runtime_data, "_async_post", side_effect=analyze):
            await async_stage_pcm_sample(hass, user.id, 0, b"\x01\x00" * 16000, 16000)
            await started.wait()
            await async_stage_pcm_sample(hass, user.id, 0, b"\x02\x00" * 16000, 16000)
            release.set()
            await hass.async_block_till_done()
            assert count == 2
            assert (
                hass.data[DOMAIN]["enrollment_quality"][user.id]["samples"][0][
                    "assessment"
                ]
                == "good"
            )
            started.clear()
            release.clear()
            await async_stage_pcm_sample(hass, user.id, 0, b"\x03\x00" * 16000, 16000)
            await started.wait()
            assert await hass.config_entries.async_unload(entry.entry_id)
            assert not hass.data[DOMAIN].get("enrollment_quality_tasks")
            assert not hass.data[DOMAIN].get("enrollment_quality")
    finally:
        release.set()
        await backend.stop()
