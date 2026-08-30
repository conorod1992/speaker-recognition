"""Tests for the streaming-safe STT DSP path."""

from array import array
import asyncio
import importlib.util
import math
from pathlib import Path
import sys


def _load_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "streaming_enhancement.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_streaming_enhancement", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()
StreamingBasicDSP = module.StreamingBasicDSP
async_enhance_stt_stream = module.async_enhance_stt_stream


def _tone(frequency: float, sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    return array(
        "h",
        (
            int(8000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(int(sample_rate * seconds))
        ),
    ).tobytes()


def _rms(pcm_data: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm_data)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _process_in_chunks(pcm_data: bytes, chunk_size: int = 317) -> bytes:
    dsp = StreamingBasicDSP(16000, 1)
    output = bytearray()
    for offset in range(0, len(pcm_data), chunk_size):
        output.extend(dsp.process(pcm_data[offset : offset + chunk_size]))
    output.extend(dsp.flush())
    return bytes(output)


def test_streaming_dsp_preserves_length_across_odd_chunk_boundaries() -> None:
    original = _tone(1000.0)
    enhanced = _process_in_chunks(original)

    assert len(enhanced) == len(original)
    assert _rms(enhanced) > 1000


def test_streaming_dsp_reduces_mains_more_than_speech_band() -> None:
    mains = _tone(50.0)
    speech = _tone(1000.0)

    mains_ratio = _rms(_process_in_chunks(mains)) / _rms(mains)
    speech_ratio = _rms(_process_in_chunks(speech)) / _rms(speech)

    assert mains_ratio < speech_ratio


def test_wav_container_is_passed_through_unchanged() -> None:
    payload = b"RIFF" + (32).to_bytes(4, "little") + b"WAVE" + b"test-payload"

    async def source():
        yield payload[:7]
        yield payload[7:13]
        yield payload[13:]

    async def collect() -> bytes:
        chunks = []
        async for chunk in async_enhance_stt_stream(source(), 16000, 1):
            chunks.append(chunk)
        return b"".join(chunks)

    assert asyncio.run(collect()) == payload
