"""Lightweight generic whisper detection for captured Assist speech."""

from __future__ import annotations

from array import array
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import statistics
import sys
from threading import Lock

_TARGET_SAMPLE_RATE = 8000
_FRAME_SECONDS = 0.032
_MAX_ANALYSIS_FRAMES = 24
_MIN_ACTIVE_FRAMES = 4
_MIN_SIGNAL_RMS = 64.0
_MIN_PITCH_HZ = 70.0
_MAX_PITCH_HZ = 350.0
_VOICED_PERIODICITY = 0.42
_STRONG_VOICED_PERIODICITY = 0.62
_WHISPER_THRESHOLD = 0.60
_SPECTRAL_BIN_HZ = 125
_DETECTION_CACHE_LIMIT = 16


@dataclass(frozen=True)
class WhisperDetection:
    """Result of generic speech-style analysis."""

    whispering: bool
    score: float
    available: bool
    periodicity: float = 0.0
    voiced_fraction: float = 0.0
    peak_periodicity: float = 0.0
    strong_voiced_fraction: float = 0.0
    normal_voicing_rescue: float = 0.0
    spectral_flatness: float = 0.0
    spectral_centroid_hz: float = 0.0
    low_frequency_ratio: float = 0.0
    high_frequency_ratio: float = 0.0
    zero_crossing_rate: float = 0.0
    difference_ratio: float = 0.0
    voicing_score: float = 0.0
    spectral_score: float = 0.0

    def diagnostics(self) -> dict[str, float]:
        """Return stable, user-facing component diagnostics."""
        return {
            "periodicity": self.periodicity,
            "voiced_fraction": self.voiced_fraction,
            "peak_periodicity": self.peak_periodicity,
            "strong_voiced_fraction": self.strong_voiced_fraction,
            "normal_voicing_rescue": self.normal_voicing_rescue,
            "spectral_flatness": self.spectral_flatness,
            "spectral_centroid_hz": self.spectral_centroid_hz,
            "low_frequency_ratio": self.low_frequency_ratio,
            "high_frequency_ratio": self.high_frequency_ratio,
            "zero_crossing_rate": self.zero_crossing_rate,
            "difference_ratio": self.difference_ratio,
            "voicing_score": self.voicing_score,
            "spectral_score": self.spectral_score,
        }


_DETECTION_CACHE: OrderedDict[tuple[bytes, int], WhisperDetection] = OrderedDict()
_DETECTION_CACHE_LOCK = Lock()


def _detection_cache_key(pcm_data: bytes, sample_rate: int) -> tuple[bytes, int]:
    """Return a compact key for one prepared PCM utterance."""
    return hashlib.blake2s(pcm_data, digest_size=12).digest(), sample_rate


def _remember_detection(
    pcm_data: bytes, sample_rate: int, detection: WhisperDetection
) -> None:
    """Keep a tiny diagnostic cache without retaining raw audio."""
    key = _detection_cache_key(pcm_data, sample_rate)
    with _DETECTION_CACHE_LOCK:
        _DETECTION_CACHE[key] = detection
        _DETECTION_CACHE.move_to_end(key)
        while len(_DETECTION_CACHE) > _DETECTION_CACHE_LIMIT:
            _DETECTION_CACHE.popitem(last=False)


def cached_detection(
    pcm_data: bytes, sample_rate: int
) -> WhisperDetection | None:
    """Return a recent result for this exact PCM utterance, if available."""
    key = _detection_cache_key(pcm_data, sample_rate)
    with _DETECTION_CACHE_LOCK:
        detection = _DETECTION_CACHE.get(key)
        if detection is not None:
            _DETECTION_CACHE.move_to_end(key)
        return detection


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


@lru_cache(maxsize=4)
def _spectral_basis(
    frame_length: int, sample_rate_tenths: int
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    """Precompute a small DFT basis reused by every frame in an utterance."""
    sample_rate = sample_rate_tenths / 10.0
    frequencies = tuple(
        float(frequency)
        for frequency in range(
            _SPECTRAL_BIN_HZ,
            int(sample_rate / 2),
            _SPECTRAL_BIN_HZ,
        )
    )
    window = tuple(
        0.5 - 0.5 * math.cos(2.0 * math.pi * index / (frame_length - 1))
        for index in range(frame_length)
    )
    cosine = tuple(
        tuple(
            math.cos(2.0 * math.pi * frequency * index / sample_rate)
            for index in range(frame_length)
        )
        for frequency in frequencies
    )
    sine = tuple(
        tuple(
            math.sin(2.0 * math.pi * frequency * index / sample_rate)
            for index in range(frame_length)
        )
        for frequency in frequencies
    )
    return frequencies, window, cosine, sine


def _frame_spectral_features(
    frame: list[float], sample_rate: float
) -> tuple[float, float, float, float]:
    """Return flatness, centroid and low/high-frequency energy ratios."""
    frequencies, window, cosine, sine = _spectral_basis(
        len(frame), round(sample_rate * 10.0)
    )
    if not frequencies:
        return 0.0, 0.0, 0.0, 0.0

    windowed = [value * weight for value, weight in zip(frame, window)]
    powers: list[float] = []
    for cosine_bin, sine_bin in zip(cosine, sine):
        real = sum(value * basis for value, basis in zip(windowed, cosine_bin))
        imaginary = sum(value * basis for value, basis in zip(windowed, sine_bin))
        powers.append(real * real + imaginary * imaginary + 1e-12)

    total_power = sum(powers)
    if total_power <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    spectral_flatness = math.exp(
        sum(math.log(power) for power in powers) / len(powers)
    ) / (total_power / len(powers))
    spectral_centroid_hz = sum(
        frequency * power for frequency, power in zip(frequencies, powers)
    ) / total_power
    low_frequency_ratio = sum(
        power
        for frequency, power in zip(frequencies, powers)
        if frequency <= 1500.0
    ) / total_power
    high_frequency_ratio = sum(
        power
        for frequency, power in zip(frequencies, powers)
        if frequency >= 2000.0
    ) / total_power

    return (
        _clamp(spectral_flatness),
        spectral_centroid_hz,
        _clamp(low_frequency_ratio),
        _clamp(high_frequency_ratio),
    )


def _evenly_spaced_frames(
    frames: list[tuple[float, list[float]]], limit: int
) -> list[tuple[float, list[float]]]:
    """Keep phonetic variety instead of retaining only the loudest frames."""
    if len(frames) <= limit:
        return frames
    if limit <= 1:
        return [frames[len(frames) // 2]]
    indexes = [
        round(index * (len(frames) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [frames[index] for index in indexes]


def _upper_percentile(values: list[float], fraction: float) -> float:
    """Return a deterministic upper percentile without adding a numeric dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round(_clamp(fraction) * (len(ordered) - 1))
    return float(ordered[index])


def detect_whisper(pcm_data: bytes, sample_rate: int) -> WhisperDetection:
    """Classify an utterance as whispered or normally voiced speech.

    Loudness is used only to discard silence. Classification combines two
    independent signal families:

    * voicing evidence from periodicity and voiced-frame prevalence;
    * spectral evidence from flatness, spectral centroid, high-frequency energy,
      and depletion of the low-frequency harmonic region.

    The detector also looks for a minority of strongly periodic frames. Soft or
    distant normal speech can have whisper-like median statistics after room and
    microphone processing while still retaining unmistakably voiced vowel frames.
    Those frames provide a conservative normal-speech rescue rather than making
    quietness itself evidence for or against whispering.
    """
    if sample_rate < _TARGET_SAMPLE_RATE:
        return WhisperDetection(False, 0.0, False)

    samples = _pcm_samples(pcm_data)
    if not samples:
        return WhisperDetection(False, 0.0, False)

    downsample_step = max(1, round(sample_rate / _TARGET_SAMPLE_RATE))
    downsampled = samples[::downsample_step]
    analysis_rate = sample_rate / downsample_step
    frame_length = max(128, int(analysis_rate * _FRAME_SECONDS))
    hop_length = max(64, frame_length // 2)

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
        maximum_mean_square * 0.025,
        _MIN_SIGNAL_RMS * _MIN_SIGNAL_RMS,
    )
    active = [item for item in frames if item[0] >= active_floor]
    active = _evenly_spaced_frames(active, _MAX_ANALYSIS_FRAMES)
    if len(active) < _MIN_ACTIVE_FRAMES:
        return WhisperDetection(False, 0.0, False)

    periodicities: list[float] = []
    zero_crossing_rates: list[float] = []
    difference_ratios: list[float] = []
    spectral_flatnesses: list[float] = []
    spectral_centroids: list[float] = []
    low_frequency_ratios: list[float] = []
    high_frequency_ratios: list[float] = []

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

        (
            flatness,
            centroid_hz,
            low_ratio,
            high_ratio,
        ) = _frame_spectral_features(frame, analysis_rate)
        spectral_flatnesses.append(flatness)
        spectral_centroids.append(centroid_hz)
        low_frequency_ratios.append(low_ratio)
        high_frequency_ratios.append(high_ratio)

    periodicity = float(statistics.median(periodicities))
    voiced_fraction = sum(
        value >= _VOICED_PERIODICITY for value in periodicities
    ) / len(periodicities)
    peak_periodicity = _upper_percentile(periodicities, 0.85)
    strong_voiced_fraction = sum(
        value >= _STRONG_VOICED_PERIODICITY for value in periodicities
    ) / len(periodicities)
    zero_crossing_rate = float(statistics.median(zero_crossing_rates))
    difference_ratio = float(statistics.median(difference_ratios))
    spectral_flatness = float(statistics.median(spectral_flatnesses))
    spectral_centroid_hz = float(statistics.median(spectral_centroids))
    low_frequency_ratio = float(statistics.median(low_frequency_ratios))
    high_frequency_ratio = float(statistics.median(high_frequency_ratios))

    periodicity_evidence = _clamp((0.55 - periodicity) / 0.35)
    voiced_fraction_evidence = _clamp((0.65 - voiced_fraction) / 0.55)
    zero_crossing_evidence = _clamp((zero_crossing_rate - 0.10) / 0.22)
    difference_evidence = _clamp((difference_ratio - 0.55) / 0.80)

    flatness_evidence = _clamp((spectral_flatness - 0.12) / 0.38)
    centroid_evidence = _clamp((spectral_centroid_hz - 900.0) / 1300.0)
    high_frequency_evidence = _clamp((high_frequency_ratio - 0.06) / 0.30)
    low_frequency_depletion = _clamp((0.82 - low_frequency_ratio) / 0.42)

    voicing_score = (
        0.55 * periodicity_evidence
        + 0.30 * voiced_fraction_evidence
        + 0.10 * zero_crossing_evidence
        + 0.05 * difference_evidence
    )
    spectral_score = (
        0.30 * flatness_evidence
        + 0.25 * centroid_evidence
        + 0.25 * high_frequency_evidence
        + 0.20 * low_frequency_depletion
    )

    peak_rescue = _clamp((peak_periodicity - 0.58) / 0.25)
    strong_fraction_rescue = _clamp((strong_voiced_fraction - 0.04) / 0.29)
    normal_voicing_rescue = _clamp(
        0.60 * peak_rescue + 0.40 * strong_fraction_rescue
    )

    agreement = min(voicing_score, spectral_score)
    raw_score = _clamp(
        0.42 * voicing_score + 0.42 * spectral_score + 0.16 * agreement
    )
    score = _clamp(raw_score - 0.22 * normal_voicing_rescue)

    detection = WhisperDetection(
        whispering=score >= _WHISPER_THRESHOLD,
        score=score,
        available=True,
        periodicity=periodicity,
        voiced_fraction=voiced_fraction,
        peak_periodicity=peak_periodicity,
        strong_voiced_fraction=strong_voiced_fraction,
        normal_voicing_rescue=normal_voicing_rescue,
        spectral_flatness=spectral_flatness,
        spectral_centroid_hz=spectral_centroid_hz,
        low_frequency_ratio=low_frequency_ratio,
        high_frequency_ratio=high_frequency_ratio,
        zero_crossing_rate=zero_crossing_rate,
        difference_ratio=difference_ratio,
        voicing_score=voicing_score,
        spectral_score=spectral_score,
    )
    _remember_detection(pcm_data, sample_rate, detection)
    return detection
