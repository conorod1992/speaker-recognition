"""Explicit, bounded staging of labelled Assist audio as managed enrollment media."""

from __future__ import annotations

import asyncio
import base64
from typing import Any

from homeassistant.helpers.storage import Store

from .calibration import _actual_user_for_engine_trial
from .const import DOMAIN
from .enrollment import _delete_paths, _managed_media_path, _sample_location, _write_wav
from .telemetry import get_decision_history

MAX_PROMOTED_PER_USER = 6
MAX_PENDING_PROMOTIONS = 24


def confirmed_user(record: dict[str, Any]) -> str | None:
    """A label, never the raw candidate, supplies the target identity."""
    if record.get("feedback") not in ("correct", "wrong_speaker", "missed_speaker"):
        return None
    return _actual_user_for_engine_trial(record)


def promotion_lock(hass):
    return hass.data.setdefault(DOMAIN, {}).setdefault("promotion_lock", asyncio.Lock())


def pending_promotions(hass, user_id: str | None = None) -> list[dict]:
    rows = hass.data.get(DOMAIN, {}).get("promoted_samples", [])
    configured = [
        sample
        for entry in hass.config_entries.async_entries(DOMAIN)
        for sample in entry.options.get("voice_samples", [])
    ]
    committed = _committed_ids(configured)
    return [
        dict(row)
        for row in rows
        if row["decision_id"] not in committed
        and (user_id is None or row["user"] == user_id)
    ]


async def async_setup_promotions(hass) -> None:
    data = hass.data.setdefault(DOMAIN, {})
    if "promotion_store" in data:
        return
    store = Store(hass, 1, f"{DOMAIN}.promoted_samples")
    saved = await store.async_load()
    rows = saved.get("samples", []) if isinstance(saved, dict) else []
    rows = rows if isinstance(rows, list) else []
    data["promoted_samples"] = [
        row
        for row in rows
        if isinstance(row, dict)
        and all(
            isinstance(row.get(key), str) and row[key]
            for key in ("user", "decision_id", "media_content_id")
        )
        and _managed_media_path(hass, row["media_content_id"]) is not None
    ][:MAX_PENDING_PROMOTIONS]
    data["promotion_store"] = store


async def _save(hass, rows) -> None:
    data = hass.data[DOMAIN]
    await data["promotion_store"].async_save({"samples": rows})
    data["promoted_samples"] = rows


def _items(row):
    samples = row.get("samples", [])
    return samples if isinstance(samples, list) else [samples]


def _committed_ids(samples) -> set[str]:
    return {
        item["decision_id"]
        for user in samples
        for item in _items(user)
        if isinstance(item, dict) and isinstance(item.get("decision_id"), str)
    }


async def async_reconcile_promotions(hass, samples) -> None:
    """Consume committed staging references; discard removed users' pending media."""
    if "promotion_store" not in hass.data.get(DOMAIN, {}):
        return
    async with promotion_lock(hass):
        users = {row.get("user") for row in samples}
        committed = _committed_ids(samples)
        rows = list(hass.data[DOMAIN].get("promoted_samples", []))
        retained = [
            row
            for row in rows
            if row["user"] in users and row["decision_id"] not in committed
        ]
        if retained != rows:
            await _save(hass, retained)
        paths = [
            _managed_media_path(hass, row["media_content_id"])
            for row in rows
            if row["user"] not in users and row["decision_id"] not in committed
        ]
        await hass.async_add_executor_job(
            _delete_paths, [path for path in paths if path is not None]
        )


async def async_promote_decision(hass, entry, decision_id: str) -> dict:
    """Copy retained audio after explicit confirmation, without touching the profile."""
    async with promotion_lock(hass):
        if any(
            not waiter.done()
            for waiter in hass.data[DOMAIN]
            .get("enrollment_commit_waiters", {})
            .values()
        ):
            raise ValueError("Training is in progress; try again after it finishes")
        history = get_decision_history(hass)
        record = (
            next(
                (
                    row
                    for row in history.labelled()
                    if row.get("decision_id") == decision_id
                ),
                None,
            )
            if history
            else None
        )
        target = confirmed_user(record) if record else None
        user = await hass.auth.async_get_user(target) if target else None
        runtime = getattr(entry, "runtime_data", None)
        if (
            not user
            or user.system_generated
            or not user.is_active
            or runtime is None
            or target not in runtime.enrolled_users
            or target not in runtime.configured_users
        ):
            raise ValueError(
                "Explicit ground truth for an active enrolled Home Assistant user is required"
            )
        samples = entry.options.get("voice_samples", [])
        pending = pending_promotions(hass)
        committed = _committed_ids(samples)
        if (
            record.get("promoted_user_id")
            or decision_id in committed
            or any(row["decision_id"] == decision_id for row in pending)
        ):
            raise ValueError("This decision has already been promoted")
        total = sum(row["user"] == target for row in pending) + sum(
            item.get("source") == "assist"
            for row in samples
            if row.get("user") == target
            for item in _items(row)
            if isinstance(item, dict)
        )
        if total >= MAX_PROMOTED_PER_USER or len(pending) >= MAX_PENDING_PROMOTIONS:
            raise ValueError("The bounded promoted-sample limit has been reached")
        clip = history.review_audio_for_decision(decision_id)
        if clip is None:
            raise ValueError("The retained audio has expired")
        pcm = base64.b64decode(clip["pcm_base64"], validate=True)
        rate = clip["sample_rate"]
        if (
            not isinstance(rate, int)
            or not 8000 <= rate <= 48000
            or not rate <= len(pcm) <= rate * 60
            or len(pcm) % 2
        ):
            raise ValueError("Invalid retained PCM audio")
        path, media_id = _sample_location(hass, target, 6 + total)
        try:
            await hass.async_add_executor_job(_write_wav, path, pcm, rate)
            # Recheck after disk I/O: a simultaneous user/profile removal must not
            # leave a new staged recording assigned to an ineligible profile.
            user = await hass.auth.async_get_user(target)
            if (
                not user
                or not user.is_active
                or user.system_generated
                or not any(
                    row.get("user") == target
                    for row in entry.options.get("voice_samples", [])
                )
            ):
                raise ValueError("The target user is no longer enrolled")
            latest = next(
                (
                    row
                    for row in history.labelled()
                    if row.get("decision_id") == decision_id
                ),
                None,
            )
            if latest is None or confirmed_user(latest) != target:
                raise ValueError("The clip label changed while it was being promoted")
            item = {
                "user": target,
                "decision_id": decision_id,
                "media_content_id": media_id,
                "source": "assist",
            }
            await _save(hass, pending + [item])
        except Exception:
            await hass.async_add_executor_job(_delete_paths, [path])
            raise
        history.mark_promoted(decision_id, target)
        from .enrollment_quality import schedule_quality_analysis

        schedule_quality_analysis(hass, target)
        return item


async def async_discard_promotions(hass, user_id: str) -> None:
    async with promotion_lock(hass):
        if any(
            not waiter.done()
            for waiter in hass.data[DOMAIN]
            .get("enrollment_commit_waiters", {})
            .values()
        ):
            raise ValueError("Training is in progress")
        rows = pending_promotions(hass)
        discarded = [row for row in rows if row["user"] == user_id]
        await _save(hass, [row for row in rows if row["user"] != user_id])
        paths = [
            _managed_media_path(hass, row["media_content_id"]) for row in discarded
        ]
        await hass.async_add_executor_job(
            _delete_paths, [path for path in paths if path is not None]
        )
        from .enrollment_quality import schedule_quality_analysis

        schedule_quality_analysis(hass, user_id)
