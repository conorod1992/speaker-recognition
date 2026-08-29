"""Dependency-free speech enhancement helpers for live satellite diagnostics."""

from __future__ import annotations

from array import array
import base64
from io import BytesIO
import math
from time import perf_counter
from typing import Any
import wave


def _clip_int16(value: float) -> int:
    """Clamp a floating-point sample to signed 16-bit PCM."""
    return max(-32768, min(32767, int(round(value))))


def _biquad_notch_coefficients(
    sample_rate: int, frequency: float, q: float = 24.0
) -> tuple[float, float, float, float, float]:
    """Return normalized biquad notch coefficients."""
    omega = 2.0 * math.pi * frequency / sample_rate
    alpha = math.sin(omega) / (2.0 * q)
    cosine = math.cos(omega)
    a0 = 1.0 + alpha
    return (
        1.0 / a0,
        (-2.0 * cosine) / a0,
        1.0 / a0,
        (-2.0 * cosine) / a0,
        (1.0 - alpha) / a0,
    )


def _apply_notch(samples: list[float], sample_rate: int, frequency: float) -> None:
    """Apply an in-place biquad notch filter."""
    if frequency <= 0 or frequency >= sample_rate / 2:
        return
    b0, b1, b2, a1, a2 = _biquad_notch_coefficients(sample_rate, frequency)
    x1 = x2 = y1 = y2 = 0.0
    for index, sample in enumerate(samples):
        output = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        samples[index] = output
        x2, x1 = x1, sample
        y2, y1 = y1, output


def _apply_high_pass(samples: list[float], sample_rate: int, cutoff: float = 85.0) -> None:
    """Apply a simple first-order high-pass filter in place."""
    if not samples:
        return
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff)
    alpha = rc / (rc + dt)
    previous_input = samples[0]
    previous_output = 0.0
    for index, sample in enumerate(samples):
        output = alpha * (previous_output + sample - previous_input)
        samples[index] = output
        previous_input = sample
        previous_output = output


def _apply_conservative_noise_attenuation(
    samples: list[float], sample_rate: int
) -> None:
    """Attenuate low-energy frames while preserving likely speech frames."""
    frame_size = max(1, sample_rate // 50)  # 20 ms
    frame_rms: list[float] = []
    for start in range(0, len(samples), frame_size):
        frame = samples[start : start + frame_size]
        if not frame:
            continue
        energy = sum(sample * sample for sample in frame) / len(frame)
        frame_rms.append(math.sqrt(energy))
    if not frame_rms:
        return

    ordered = sorted(frame_rms)
    noise_index = min(len(ordered) - 1, max(0, len(ordered) // 5))
    noise_floor = max(20.0, ordered[noise_index])
    low_threshold = noise_floor * 1.6
    high_threshold = noise_floor * 3.2
    smoothed_gain = 1.0

    for frame_index, start in enumerate(range(0, len(samples), frame_size)):
        rms = frame_rms[min(frame_index, len(frame_rms) - 1)]
        if rms <= low_threshold:
            target_gain = 0.35
        elif rms >= high_threshold:
            target_gain = 1.0
        else:
            fraction = (rms - low_threshold) / (high_threshold - low_threshold)
            target_gain = 0.35 + (0.65 * fraction)
        smoothed_gain = (0.72 * smoothed_gain) + (0.28 * target_gain)
        end = min(len(samples), start + frame_size)
        for index in range(start, end):
            samples[index] *= smoothed_gain


def enhance_speech_pcm(pcm_data: bytes, sample_rate: int) -> bytes:
    """Apply conservative speech-focused cleanup to mono signed 16-bit PCM."""
    if sample_rate <= 0 or len(pcm_data) < 2 or len(pcm_data) % 2:
        return pcm_data

    pcm = array("h")
    pcm.frombytes(pcm_data)
    if pcm.itemsize != 2 or not pcm:
        return pcm_data

    mean = sum(pcm) / len(pcm)
    samples = [float(sample) - mean for sample in pcm]

    _apply_high_pass(samples, sample_rate)
    _apply_notch(samples, sample_rate, 50.0)
    _apply_notch(samples, sample_rate, 100.0)
    _apply_conservative_noise_attenuation(samples, sample_rate)

    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak > 0:
        gain = min(3.0, (32767.0 * 0.90) / peak)
        samples = [sample * gain for sample in samples]

    output = array("h", (_clip_int16(sample) for sample in samples))
    return output.tobytes()


def wav_base64(pcm_data: bytes, sample_rate: int) -> str:
    """Encode mono signed 16-bit PCM as a base64 WAV payload."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_comparison_preview(
    original_pcm: bytes,
    basic_pcm: bytes,
    sample_rate: int,
    basic_processing_seconds: float,
    neural_pcm: bytes | None = None,
    neural_processing_seconds: float | None = None,
    neural_engine: str | None = None,
    neural_error: str | None = None,
) -> dict[str, Any]:
    """Build an A/B/C preview payload from precomputed enhancement stages."""
    result: dict[str, Any] = {
        "sample_rate": sample_rate,
        "audio_seconds": len(original_pcm) / (sample_rate * 2),
        "processing_seconds": basic_processing_seconds,
        "basic_processing_seconds": basic_processing_seconds,
        "original_wav_base64": wav_base64(original_pcm, sample_rate),
        "enhanced_wav_base64": wav_base64(basic_pcm, sample_rate),
    }
    if neural_pcm is not None:
        result["neural_wav_base64"] = wav_base64(neural_pcm, sample_rate)
        result["neural_processing_seconds"] = neural_processing_seconds or 0.0
        result["neural_engine"] = neural_engine or "rnnoise"
    if neural_error:
        result["neural_error"] = neural_error
    return result


def build_enhancement_preview(pcm_data: bytes, sample_rate: int) -> dict[str, Any]:
    """Build original/basic WAV previews and processing timing."""
    started = perf_counter()
    enhanced = enhance_speech_pcm(pcm_data, sample_rate)
    processing_seconds = perf_counter() - started
    return build_comparison_preview(
        pcm_data,
        enhanced,
        sample_rate,
        processing_seconds,
    )
