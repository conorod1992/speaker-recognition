"""Optional RNNoise speech denoising for the Speaker Recognition backend."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly  # type: ignore[import-untyped]

RNNOISE_SAMPLE_RATE = 48000


class NeuralDenoiseUnavailable(RuntimeError):
    """Raised when the optional RNNoise runtime is unavailable."""


def _resample_int16(
    samples: NDArray[np.int16], source_rate: int, target_rate: int
) -> NDArray[np.int16]:
    """Resample mono signed 16-bit PCM while preserving bounded sample values."""
    if source_rate == target_rate:
        return samples.copy()
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    resampled = resample_poly(samples.astype(np.float32), up, down)
    converted = np.asarray(
        np.clip(np.rint(resampled), -32768, 32767), dtype=np.int16
    )
    return converted


def denoise_pcm_rnnoise(pcm_data: bytes, sample_rate: int) -> tuple[bytes, float]:
    """Denoise mono PCM16 using RNNoise and return audio plus processing time."""
    if sample_rate <= 0 or len(pcm_data) < 2 or len(pcm_data) % 2:
        raise ValueError("Audio must be non-empty mono signed 16-bit PCM")

    try:
        from pyrnnoise.rnnoise import (  # type: ignore[import-not-found]
            FRAME_SIZE,
            create,
            destroy,
            process_mono_frame,
        )
    except (ImportError, OSError) as error:
        raise NeuralDenoiseUnavailable("RNNoise is not installed in this backend") from error

    original = np.frombuffer(pcm_data, dtype=np.int16).copy()
    if original.size == 0:
        raise ValueError("Audio must contain samples")

    started = perf_counter()
    working = _resample_int16(original, sample_rate, RNNOISE_SAMPLE_RATE)
    state = create()
    if not state:
        raise NeuralDenoiseUnavailable("RNNoise could not create a denoising state")

    frames: list[NDArray[np.int16]] = []
    try:
        for start in range(0, working.size, FRAME_SIZE):
            frame = working[start : start + FRAME_SIZE]
            denoised, _speech_probability = process_mono_frame(state, frame)
            frames.append(np.asarray(denoised, dtype=np.int16))
    finally:
        destroy(state)

    if not frames:
        raise ValueError("RNNoise did not produce any audio")

    denoised_48k = np.concatenate(frames)
    restored = _resample_int16(denoised_48k, RNNOISE_SAMPLE_RATE, sample_rate)

    # Round-trip resampling can be off by a sample or two. Match the exact live
    # utterance length so the three diagnostic players remain directly comparable.
    if restored.size > original.size:
        restored = restored[: original.size]
    elif restored.size < original.size:
        restored = np.pad(restored, (0, original.size - restored.size))

    return restored.astype(np.int16, copy=False).tobytes(), perf_counter() - started
