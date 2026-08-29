"""Focused tests for generic whisper detection on Assist PCM audio."""

from array import array
import importlib.util
import math
from pathlib import Path
import random
import sys


def _load_whisper_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "whisper.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_integration_whisper", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pcm_bytes(samples: list[int]) -> bytes:
    values = array("h", samples)
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _voiced_samples(amplitude: int, *, sample_rate: int = 16000) -> list[int]:
    """Create deterministic harmonic speech-like voiced audio."""
    sample_count = int(sample_rate * 1.5)
    return [
        int(
            amplitude
            * (
                0.65 * math.sin(2 * math.pi * 130 * index / sample_rate)
                + 0.22 * math.sin(2 * math.pi * 260 * index / sample_rate)
                + 0.10 * math.sin(2 * math.pi * 390 * index / sample_rate)
            )
        )
        for index in range(sample_count)
    ]


def _breathy_soft_voiced_samples(*, sample_rate: int = 16000) -> list[int]:
    """Create soft voiced speech with enough noise to confuse median-only features."""
    rng = random.Random(31)
    voiced = _voiced_samples(1900, sample_rate=sample_rate)
    samples = []
    for index, value in enumerate(voiced):
        envelope = 0.70 + 0.30 * math.sin(math.pi * index / len(voiced)) ** 2
        noise = rng.uniform(-900.0, 900.0)
        samples.append(int((value + noise) * envelope))
    return samples


def _whisper_like_samples(*, sample_rate: int = 16000) -> list[int]:
    """Create deterministic aperiodic, noise-excited speech-like audio."""
    rng = random.Random(17)
    sample_count = int(sample_rate * 1.5)
    previous = 0.0
    samples = []
    for index in range(sample_count):
        current = rng.uniform(-1.0, 1.0)
        noise = current - 0.55 * previous
        previous = current
        envelope = 0.65 + 0.35 * math.sin(math.pi * index / sample_count) ** 2
        value = max(-1.0, min(1.0, noise / 1.55))
        samples.append(int(value * 2500 * envelope))
    return samples


def _lowpassed_whisper_like_samples(*, sample_rate: int = 16000) -> list[int]:
    """Create whisper-like excitation with reduced high-frequency energy."""
    rng = random.Random(22)
    sample_count = int(sample_rate * 1.5)
    filtered = 0.0
    samples = []
    for index in range(sample_count):
        filtered = 0.72 * filtered + 0.28 * rng.uniform(-1.0, 1.0)
        envelope = 0.55 + 0.45 * math.sin(math.pi * index / sample_count) ** 2
        samples.append(int(filtered * 5000 * envelope))
    return samples


def test_normally_voiced_speech_is_not_classified_as_whispering() -> None:
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(_pcm_bytes(_voiced_samples(8000)), 16000)
    assert result.available
    assert not result.whispering
    assert result.voiced_fraction > 0.8
    assert result.periodicity > 0.8
    assert result.spectral_score < 0.2


def test_quiet_normally_voiced_speech_is_not_mistaken_for_a_whisper() -> None:
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(_pcm_bytes(_voiced_samples(800)), 16000)
    assert result.available
    assert not result.whispering
    assert result.voiced_fraction > 0.8
    assert result.score < 0.2


def test_breathy_soft_voice_gets_normal_voicing_rescue() -> None:
    """Strong harmonic vowel frames can rescue otherwise whisper-like soft speech."""
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(_pcm_bytes(_breathy_soft_voiced_samples()), 16000)
    assert result.available
    assert result.peak_periodicity > 0.62
    assert result.strong_voiced_fraction > 0.0
    assert result.normal_voicing_rescue > 0.0
    assert not result.whispering


def test_aperiodic_noise_excited_speech_is_classified_as_whisper_like() -> None:
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(_pcm_bytes(_whisper_like_samples()), 16000)
    assert result.available
    assert result.whispering
    assert result.voiced_fraction < 0.5
    assert result.spectral_flatness > 0.2
    assert result.spectral_score > 0.5


def test_lowpassed_whisper_still_has_enough_cross_family_evidence() -> None:
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(
        _pcm_bytes(_lowpassed_whisper_like_samples()), 16000
    )
    assert result.available
    assert result.whispering
    assert result.voicing_score > 0.7
    assert result.spectral_score > 0.35


def test_whisper_score_ranks_clear_whisper_above_normal_voice() -> None:
    whisper = _load_whisper_module()
    normal = whisper.detect_whisper(_pcm_bytes(_voiced_samples(5000)), 16000)
    whispered = whisper.detect_whisper(_pcm_bytes(_whisper_like_samples()), 16000)
    assert whispered.score > normal.score + 0.35


def test_component_diagnostics_are_cached_without_raw_audio() -> None:
    whisper = _load_whisper_module()
    pcm_data = _pcm_bytes(_whisper_like_samples())
    result = whisper.detect_whisper(pcm_data, 16000)
    cached = whisper.cached_detection(pcm_data, 16000)
    assert cached == result
    assert set(result.diagnostics()) == {
        "periodicity",
        "voiced_fraction",
        "peak_periodicity",
        "strong_voiced_fraction",
        "normal_voicing_rescue",
        "spectral_flatness",
        "spectral_centroid_hz",
        "low_frequency_ratio",
        "high_frequency_ratio",
        "zero_crossing_rate",
        "difference_ratio",
        "voicing_score",
        "spectral_score",
    }


def test_silence_does_not_produce_a_whisper_decision() -> None:
    whisper = _load_whisper_module()
    result = whisper.detect_whisper(_pcm_bytes([0] * 16000), 16000)
    assert not result.available
    assert not result.whispering
    assert result.score == 0.0
