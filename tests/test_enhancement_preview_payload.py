"""Tests for composing live audio comparison payloads."""

import base64
import importlib.util
from pathlib import Path
import wave
from io import BytesIO


def _load_enhancement_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "enhancement.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_enhancement_payload_test", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pcm() -> bytes:
    return (b"\x00\x00\x10\x00\xf0\xff") * 100


def test_three_way_preview_keeps_all_audio_at_original_rate() -> None:
    module = _load_enhancement_module()
    original = _pcm()
    basic = original
    neural = original

    result = module.build_comparison_preview(
        original,
        basic,
        16000,
        0.004,
        neural,
        0.012,
        "rnnoise",
    )

    assert result["basic_processing_seconds"] == 0.004
    assert result["neural_processing_seconds"] == 0.012
    assert result["neural_engine"] == "rnnoise"
    for key in ("original_wav_base64", "enhanced_wav_base64", "neural_wav_base64"):
        wav_bytes = base64.b64decode(result[key])
        with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
            assert wav_file.getframerate() == 16000
            assert wav_file.getnchannels() == 1
            assert wav_file.getsampwidth() == 2


def test_neural_failure_keeps_original_and_basic_preview() -> None:
    module = _load_enhancement_module()
    result = module.build_comparison_preview(
        _pcm(), _pcm(), 16000, 0.003, neural_error="backend unavailable"
    )

    assert "original_wav_base64" in result
    assert "enhanced_wav_base64" in result
    assert "neural_wav_base64" not in result
    assert result["neural_error"] == "backend unavailable"
