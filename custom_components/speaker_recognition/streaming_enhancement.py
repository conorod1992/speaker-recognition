"""Streaming-safe basic speech enhancement for wrapped STT audio."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
import math


@dataclass
class _FilterState:
    """Per-channel state for the causal filters."""

    hp_previous_input: float = 0.0
    hp_previous_output: float = 0.0
    notch_50_x1: float = 0.0
    notch_50_x2: float = 0.0
    notch_50_y1: float = 0.0
    notch_50_y2: float = 0.0
    notch_100_x1: float = 0.0
    notch_100_x2: float = 0.0
    notch_100_y1: float = 0.0
    notch_100_y2: float = 0.0


def _clip_int16(value: float) -> int:
    return max(-32768, min(32767, int(round(value))))


def _notch_coefficients(
    sample_rate: int, frequency: float, q: float = 24.0
) -> tuple[float, float, float, float, float]:
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


class StreamingBasicDSP:
    """Causal approximation of the diagnostic basic DSP path.

    The diagnostic preview can inspect a complete utterance before processing it.
    Production STT cannot do that without delaying transcription until speech ends,
    so this class keeps filter state across chunks and uses a rolling noise estimate.
    It intentionally omits whole-utterance peak normalization.
    """

    def __init__(self, sample_rate: int, channels: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self._sample_rate = sample_rate
        self._channels = channels
        self._states = [_FilterState() for _ in range(channels)]
        self._frame_samples = max(1, sample_rate // 50) * channels  # 20 ms
        self._filtered_pending: list[float] = []
        self._byte_pending = b""
        self._recent_rms: deque[float] = deque(maxlen=50)
        self._smoothed_gain = 1.0

        dt = 1.0 / sample_rate
        rc = 1.0 / (2.0 * math.pi * 85.0)
        self._hp_alpha = rc / (rc + dt)
        self._notch_50 = _notch_coefficients(sample_rate, 50.0)
        self._notch_100 = _notch_coefficients(sample_rate, 100.0)

    @staticmethod
    def _apply_notch(
        sample: float,
        coefficients: tuple[float, float, float, float, float],
        state: _FilterState,
        prefix: str,
    ) -> float:
        b0, b1, b2, a1, a2 = coefficients
        x1 = getattr(state, f"{prefix}_x1")
        x2 = getattr(state, f"{prefix}_x2")
        y1 = getattr(state, f"{prefix}_y1")
        y2 = getattr(state, f"{prefix}_y2")
        output = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        setattr(state, f"{prefix}_x2", x1)
        setattr(state, f"{prefix}_x1", sample)
        setattr(state, f"{prefix}_y2", y1)
        setattr(state, f"{prefix}_y1", output)
        return output

    def _filter_sample(self, sample: float, channel: int) -> float:
        state = self._states[channel]
        high_passed = self._hp_alpha * (
            state.hp_previous_output + sample - state.hp_previous_input
        )
        state.hp_previous_input = sample
        state.hp_previous_output = high_passed
        filtered = self._apply_notch(
            high_passed, self._notch_50, state, "notch_50"
        )
        return self._apply_notch(filtered, self._notch_100, state, "notch_100")

    def _attenuate_frame(self, frame: list[float]) -> bytes:
        if not frame:
            return b""
        rms = math.sqrt(sum(sample * sample for sample in frame) / len(frame))
        self._recent_rms.append(rms)

        target_gain = 1.0
        if len(self._recent_rms) >= 10:
            ordered = sorted(self._recent_rms)
            noise_index = min(len(ordered) - 1, max(0, len(ordered) // 5))
            noise_floor = max(20.0, ordered[noise_index])
            low_threshold = noise_floor * 1.6
            high_threshold = noise_floor * 3.2
            if rms <= low_threshold:
                target_gain = 0.35
            elif rms < high_threshold:
                fraction = (rms - low_threshold) / (high_threshold - low_threshold)
                target_gain = 0.35 + (0.65 * fraction)

        self._smoothed_gain = (0.72 * self._smoothed_gain) + (
            0.28 * target_gain
        )
        output = bytearray()
        for sample in frame:
            output.extend(
                _clip_int16(sample * self._smoothed_gain).to_bytes(
                    2, "little", signed=True
                )
            )
        return bytes(output)

    def process(self, chunk: bytes) -> bytes:
        """Process one raw interleaved PCM16 chunk."""
        data = self._byte_pending + chunk
        frame_bytes = self._channels * 2
        complete_length = len(data) - (len(data) % frame_bytes)
        complete = data[:complete_length]
        self._byte_pending = data[complete_length:]

        for offset in range(0, len(complete), 2):
            sample_index = offset // 2
            channel = sample_index % self._channels
            sample = int.from_bytes(complete[offset : offset + 2], "little", signed=True)
            self._filtered_pending.append(self._filter_sample(float(sample), channel))

        output = bytearray()
        while len(self._filtered_pending) >= self._frame_samples:
            frame = self._filtered_pending[: self._frame_samples]
            del self._filtered_pending[: self._frame_samples]
            output.extend(self._attenuate_frame(frame))
        return bytes(output)

    def flush(self) -> bytes:
        """Flush the final partial DSP frame without dropping input bytes."""
        output = bytearray()
        if self._filtered_pending:
            output.extend(self._attenuate_frame(self._filtered_pending))
            self._filtered_pending = []
        if self._byte_pending:
            output.extend(self._byte_pending)
            self._byte_pending = b""
        return bytes(output)


async def async_enhance_stt_stream(
    stream: AsyncIterable[bytes], sample_rate: int, channels: int
) -> AsyncIterator[bytes]:
    """Apply streaming DSP to raw PCM while preserving WAV containers safely.

    Home Assistant can describe a stream as WAV while yielding either raw PCM or a
    complete WAV container. A WAV header cannot be filtered as audio samples, so a
    detected container is passed through unchanged. Normal Assist raw PCM receives
    the causal DSP path without waiting for the complete utterance.
    """
    prefix = bytearray()
    passthrough = False
    processor = StreamingBasicDSP(sample_rate, channels)

    async for chunk in stream:
        if passthrough:
            yield chunk
            continue

        if len(prefix) < 12:
            prefix.extend(chunk)
            if len(prefix) < 12:
                continue
            if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
                passthrough = True
                yield bytes(prefix)
                prefix.clear()
                continue
            chunk = bytes(prefix)
            prefix.clear()

        processed = processor.process(chunk)
        if processed:
            yield processed

    if passthrough:
        return
    if prefix:
        processed = processor.process(bytes(prefix))
        if processed:
            yield processed
    final = processor.flush()
    if final:
        yield final
