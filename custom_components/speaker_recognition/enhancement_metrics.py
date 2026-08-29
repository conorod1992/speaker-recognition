"""Objective metrics for diagnostic speech-enhancement comparisons."""

from __future__ import annotations

from array import array
import math
from typing import Any


def _dbfs(rms: float) -> float:
    """Convert signed 16-bit PCM RMS to dBFS with a practical floor."""
    if rms <= 0:
        return -96.0
    return max(-96.0, 20.0 * math.log10(rms / 32768.0))


def audio_quality_metrics(pcm_data: bytes, sample_rate: int) -> dict[str, Any]:
    """Estimate quiet-frame noise floor and speech-to-noise separation."""
    if sample_rate <= 0 or len(pcm_data) < 2 or len(pcm_data) % 2:
        return {}

    pcm = array("h")
    pcm.frombytes(pcm_data)
    if pcm.itemsize != 2 or not pcm:
        return {}

    frame_size = max(1, sample_rate // 50)  # 20 ms
    frame_rms: list[float] = []
    for start in range(0, len(pcm), frame_size):
        frame = pcm[start : start + frame_size]
        if not frame:
            continue
        energy = sum(float(sample) * float(sample) for sample in frame) / len(frame)
        frame_rms.append(math.sqrt(energy))

    if not frame_rms:
        return {}

    ordered = sorted(frame_rms)
    quiet_count = max(1, len(ordered) // 5)
    loud_count = max(1, len(ordered) * 3 // 10)
    noise_rms = math.sqrt(sum(value * value for value in ordered[:quiet_count]) / quiet_count)
    speech_rms = math.sqrt(sum(value * value for value in ordered[-loud_count:]) / loud_count)
    noise_dbfs = _dbfs(noise_rms)
    speech_dbfs = _dbfs(speech_rms)

    return {
        "noise_floor_dbfs": round(noise_dbfs, 1),
        "speech_level_dbfs": round(speech_dbfs, 1),
        "estimated_snr_db": round(max(0.0, speech_dbfs - noise_dbfs), 1),
    }
