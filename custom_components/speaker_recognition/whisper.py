"""Lightweight generic whisper detection for captured Assist speech."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import statistics
import sys

_TARGET_SAMPLE_RATE = 4000
_FRAME_SECONDS = 0.03
_MAX_ANALYSIS_FRAMES = 24
_MIN_ACTIVE_FRAMES = 3
_MIN_SIGNAL_RMS = 64.0
_MIN_PITCH_HZ = 70.0
_MAX_PITCH_HZ = 350.0
_VOICED_PERIODICITY = 0.38
_WHISPER_THRESHOLD = 0.62


@dataclass(frozen=True)
class WhisperDetection:
    """Result of generic speech-style analysis."""

    whispering: bool
    score: float
    available: bool
    periodicity: float = 0.0
    voiced_fraction: float = 0.0


def _clamp(value: float) -> float:
    """Clamp a numeric score to the inclusive zero-to-one range."""
    return max(0.0, min(1.0, value))


def _pcm_samples(pcm_data: bytes) -> list[int]:
    """Decode little-endian signed 16-bit PCM into host-native integers."""
    if not pcm_data or len(pcm_data) % 2:
        return []

    values = array("h")
    values.frombytes(pcm_data)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def _frame_periodicity(frame: list[float], sample_rate: float) -> float:
    """Return the strongest normalized autocorrelation in the speech pitch range."""
    min_lag = max(2, int(sample_rate / _MAX_PITCH_HZ))
    max_lag = min(len(frame) // 2, int(sample_rate / _MIN_PITCH_HZ))
    if max_lag <= min_lag:
        return 0.0

    best = 0.0
    for lag in range(min_lag, max_lag + 1):
        left = frame[:-lag]
        right = frame[lag:]
        left_energy = sum(value * value for value in left)
        right_energy = sum(value * value for value in right)
        if left_energy <= 0.0 or right_energy <= 0.0:
            continue
        correlation = sum(a * b for a, b in zip(left, right))
        correlation /= math.sqrt(left_energy * right_energy)
        best = max(best, correlation)
    return _clamp(best)


def detect_whisper(pcm_data: bytes, sample_rate: int) -> WhisperDetection:
    """Classify an utterance as whispered or normally voiced speech.

    Loudness is used only to discard silence. The classification itself is based
    mainly on loss of periodic voicing, with zero-crossing and first-difference
    energy as supporting evidence for the noise-like excitation of whispering.
    """
    if sample_rate < 4000:
        return WhisperDetection(False, 0.0, False)

    samples = _pcm_samples(pcm_data)
    if not samples:
        return WhisperDetection(False, 0.0, False)

    downsample_step = max(1, round(sample_rate / _TARGET_SAMPLE_RATE))
    downsampled = samples[::downsample_step]
    analysis_rate = sample_rate / downsample_step
    frame_length = max(80, int(analysis_rate * _FRAME_SECONDS))
    hop_length = max(40, frame_length // 2)

    frames: list[tuple[float, list[float]]] = []
    for start in range(0, len(downsampled) - frame_length + 1, hop_length):
        raw_frame = downsampled[start : start + frame_length]
        mean = sum(raw_frame) / len(raw_frame)
        frame = [float(value) - mean for value in raw_frame]
        mean_square = sum(value * value for value in frame) / len(frame)
        frames.append((mean_square, frame))

    if not frames:
        return WhisperDetection(False, 0.0, False)

    maximum_mean_square = max(energy for energy, _ in frames)
    active_floor = max(
        maximum_mean_square * 0.04,
        _MIN_SIGNAL_RMS * _MIN_SIGNAL_RMS,
    )
    active = [item for item in frames if item[0] >= active_floor]
    active.sort(key=lambda item: item[0], reverse=True)
    active = active[:_MAX_ANALYSIS_FRAMES]
    if len(active) < _MIN_ACTIVE_FRAMES:
        return WhisperDetection(False, 0.0, False)

    periodicities: list[float] = []
    zero_crossing_rates: list[float] = []
    difference_ratios: list[float] = []

    for _, frame in active:
        periodicities.append(_frame_periodicity(frame, analysis_rate))
        zero_crossings = sum(
            (frame[index] >= 0.0) != (frame[index - 1] >= 0.0)
            for index in range(1, len(frame))
        )
        zero_crossing_rates.append(zero_crossings / (len(frame) - 1))

        signal_energy = sum(value * value for value in frame)
        difference_energy = sum(
            (frame[index] - frame[index - 1]) ** 2
            for index in range(1, len(frame))
        )
        difference_ratios.append(
            math.sqrt(difference_energy / max(signal_energy, 1.0))
        )

    periodicity = float(statistics.median(periodicities))
    voiced_fraction = sum(
        value >= _VOICED_PERIODICITY for value in periodicities
    ) / len(periodicities)
    zero_crossing_rate = float(statistics.median(zero_crossing_rates))
    difference_ratio = float(statistics.median(difference_ratios))

    periodicity_evidence = _clamp((0.38 - periodicity) / 0.25)
    voicing_evidence = _clamp((0.50 - voiced_fraction) / 0.40)
    zero_crossing_evidence = _clamp((zero_crossing_rate - 0.08) / 0.16)
    difference_evidence = _clamp((difference_ratio - 0.55) / 0.75)
    noise_evidence = (zero_crossing_evidence + difference_evidence) / 2.0

    score = (
        0.58 * periodicity_evidence
        + 0.30 * voicing_evidence
        + 0.12 * noise_evidence
    )
    score = _clamp(score)
    return WhisperDetection(
        whispering=score >= _WHISPER_THRESHOLD,
        score=score,
        available=True,
        periodicity=periodicity,
        voiced_fraction=voiced_fraction,
    )
