"""On-capture, advisory analysis of staged enrollment recordings."""

from __future__ import annotations

import asyncio
import base64

from .audio import decode_persisted_training_wav, read_bounded_wav
from .const import DOMAIN
from .enrollment import _managed_media_path


def _snapshot(hass, user_id):
    staged = hass.data.get(DOMAIN, {}).get("enrollment_staged", {}).get(user_id, {})
    from .promotions import pending_promotions

    phrases = tuple(
        (index, item["media_content_id"]) for index, item in sorted(staged.items())
    )
    promoted = tuple(
        (6 + index, item["media_content_id"])
        for index, item in enumerate(pending_promotions(hass, user_id))
    )
    return phrases + promoted


def schedule_quality_analysis(hass, user_id):
    """Coalesce captures into one finite task per user; no periodic ML jobs."""
    data = hass.data.setdefault(DOMAIN, {})
    runtime = next(
        (
            getattr(entry, "runtime_data", None)
            for entry in hass.config_entries.async_entries(DOMAIN)
            if entry.data.get("entry_type", "main") == "main"
        ),
        None,
    )
    quality = data.setdefault("enrollment_quality", {})
    if runtime is None:
        quality[user_id] = {"state": "unavailable"}
        return
    quality[user_id] = {"state": "analyzing"}
    tasks = data.setdefault("enrollment_quality_tasks", {})
    if user_id not in tasks or tasks[user_id].done():
        tasks[user_id] = hass.async_create_task(_analyze(hass, runtime, user_id))


async def _analyze(hass, runtime, user_id):
    data = hass.data[DOMAIN]
    try:
        while snapshot := _snapshot(hass, user_id):
            try:
                user = await hass.auth.async_get_user(user_id)
                if not user or user.system_generated or not user.is_active:
                    raise ValueError("Invalid enrollment user")
                samples = []
                for _, media_id in snapshot:
                    path = _managed_media_path(hass, media_id)
                    if path is None:
                        raise ValueError("Sample is not managed enrollment audio")
                    wav = await hass.async_add_executor_job(read_bounded_wav, path)
                    pcm, rate = await hass.async_add_executor_job(
                        decode_persisted_training_wav, wav
                    )
                    samples.append(
                        {
                            "user": user_id,
                            "audio": {
                                "audio_data": base64.b64encode(pcm).decode("ascii"),
                                "sample_rate": rate,
                            },
                        }
                    )
                result = await runtime._async_post(
                    "/enrollment/quality",
                    {"voice_samples": samples},
                    timeout_seconds=20,
                )
                assessments = result.get("samples")
                if not isinstance(assessments, list) or len(assessments) != len(
                    snapshot
                ):
                    raise ValueError("Invalid quality response")
                mapped = {}
                for position, ((index, _), item) in enumerate(
                    zip(snapshot, assessments), 1
                ):
                    if (
                        not isinstance(item, dict)
                        or item.get("sample_index") != position
                        or item.get("assessment")
                        not in ("good", "inconsistent", "insufficient_evidence")
                    ):
                        raise ValueError("Invalid sample assessment")
                    mapped[index] = dict(item)
                quality = {
                    "state": "ready",
                    "samples": mapped,
                    "consistency": result.get("consistency"),
                }
            except Exception:
                # Advisory failures never remove a recording or block final training.
                quality = {"state": "unavailable"}
            if _snapshot(hass, user_id) == snapshot:
                data["enrollment_quality"][user_id] = quality
                return
        data.get("enrollment_quality", {}).pop(user_id, None)
    finally:
        tasks = data.get("enrollment_quality_tasks", {})
        if tasks.get(user_id) is asyncio.current_task():
            tasks.pop(user_id, None)


async def async_cancel_quality_analysis(hass):
    data = hass.data.get(DOMAIN, {})
    tasks = list(data.get("enrollment_quality_tasks", {}).values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    data.pop("enrollment_quality", None)
