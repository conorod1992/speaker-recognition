"""Optional RNNoise speech denoising for the Speaker Recognition backend."""

from __future__ import annotations

import ctypes
from importlib import metadata
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly  # type: ignore[import-untyped]

RNNOISE_SAMPLE_RATE = 48000
RNNOISE_EXPECTED_FRAME_SIZE = 480


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


def _rnnoise_library_path() -> Path:
    """Locate pyrnnoise's bundled native library without importing its package."""
    try:
        distribution = metadata.distribution("pyrnnoise")
    except metadata.PackageNotFoundError as error:
        raise NeuralDenoiseUnavailable("RNNoise is not installed in this backend") from error

    for file in distribution.files or ():
        if str(file).replace("\\", "/").endswith("pyrnnoise/librnnoise.so"):
            path = Path(distribution.locate_file(file))
            if path.is_file():
                return path
    raise NeuralDenoiseUnavailable("RNNoise native library was not found")


def _load_rnnoise() -> ctypes.CDLL:
    """Load and configure the bundled RNNoise C API."""
    try:
        library = ctypes.CDLL(str(_rnnoise_library_path()))
    except OSError as error:
        raise NeuralDenoiseUnavailable("RNNoise native library could not be loaded") from error

    library.rnnoise_create.argtypes = [ctypes.c_void_p]
    library.rnnoise_create.restype = ctypes.c_void_p
    library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    library.rnnoise_get_frame_size.restype = ctypes.c_int
    library.rnnoise_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.rnnoise_process_frame.restype = ctypes.c_float
    return library


def rnnoise_frame_size() -> int:
    """Return the native RNNoise frame size, validating the shipped runtime."""
    frame_size = int(_load_rnnoise().rnnoise_get_frame_size())
    if frame_size <= 0:
        raise NeuralDenoiseUnavailable("RNNoise returned an invalid frame size")
    return frame_size


def _process_frame(
    library: ctypes.CDLL,
    state: int,
    frame: NDArray[np.int16],
    frame_size: int,
) -> NDArray[np.int16]:
    """Process one mono PCM frame using the RNNoise C API."""
    original_size = frame.size
    working = frame.astype(np.float32)
    if original_size < frame_size:
        working = np.pad(working, (0, frame_size - original_size)).astype(
            np.float32, copy=False
        )
    pointer = working.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    library.rnnoise_process_frame(state, pointer, pointer)
    return np.asarray(working[:original_size], dtype=np.int16)


def denoise_pcm_rnnoise(pcm_data: bytes, sample_rate: int) -> tuple[bytes, float]:
    """Denoise mono PCM16 using RNNoise and return audio plus processing time."""
    if sample_rate <= 0 or len(pcm_data) < 2 or len(pcm_data) % 2:
        raise ValueError("Audio must be non-empty mono signed 16-bit PCM")

    original = np.frombuffer(pcm_data, dtype=np.int16).copy()
    if original.size == 0:
        raise ValueError("Audio must contain samples")

    started = perf_counter()
    library = _load_rnnoise()
    frame_size = int(library.rnnoise_get_frame_size())
    if frame_size != RNNOISE_EXPECTED_FRAME_SIZE:
        raise NeuralDenoiseUnavailable(
            f"Unexpected RNNoise frame size {frame_size}; expected {RNNOISE_EXPECTED_FRAME_SIZE}"
        )

    working = _resample_int16(original, sample_rate, RNNOISE_SAMPLE_RATE)
    state = library.rnnoise_create(None)
    if not state:
        raise NeuralDenoiseUnavailable("RNNoise could not create a denoising state")

    frames: list[NDArray[np.int16]] = []
    try:
        for start in range(0, working.size, frame_size):
            frame = working[start : start + frame_size]
            frames.append(_process_frame(library, state, frame, frame_size))
    finally:
        library.rnnoise_destroy(state)

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
