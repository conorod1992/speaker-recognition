"""Tests for objective live enhancement comparison metrics."""

from __future__ import annotations

from array import array
import importlib.util
from pathlib import Path


def _load_metrics_module():
    path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "enhancement_metrics.py"
    )
    spec = importlib.util.spec_from_file_location("speaker_recognition_metrics_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pcm(quiet_level: int, speech_level: int, sample_rate: int = 16000) -> bytes:
    frame_size = sample_rate // 50
    samples = array("h")
    for frame_index in range(100):
        level = quiet_level if frame_index < 40 else speech_level
        for sample_index in range(frame_size):
            samples.append(level if sample_index % 2 == 0 else -level)
    return samples.tobytes()


def test_lower_quiet_noise_improves_estimated_snr() -> None:
    metrics = _load_metrics_module()

    noisy = metrics.audio_quality_metrics(_pcm(1000, 6000), 16000)
    cleaner = metrics.audio_quality_metrics(_pcm(250, 6000), 16000)

    assert cleaner["noise_floor_dbfs"] < noisy["noise_floor_dbfs"]
    assert cleaner["estimated_snr_db"] > noisy["estimated_snr_db"]
    assert cleaner["speech_level_dbfs"] == noisy["speech_level_dbfs"]


def test_invalid_audio_returns_no_metrics() -> None:
    metrics = _load_metrics_module()

    assert metrics.audio_quality_metrics(b"", 16000) == {}
    assert metrics.audio_quality_metrics(b"\x00", 16000) == {}
    assert metrics.audio_quality_metrics(b"\x00\x00", 0) == {}
