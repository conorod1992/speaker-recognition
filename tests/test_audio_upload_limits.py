"""Regression tests for uploaded WAV duration validation."""

from io import BytesIO
import importlib.util
from pathlib import Path
import wave

import pytest


def _load_audio_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "audio.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_integration_audio", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav(seconds: int, sample_rate: int = 8_000) -> bytes:
    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_rate * seconds)
    return output.getvalue()


def test_uploaded_wav_rejects_more_than_thirty_seconds() -> None:
    audio = _load_audio_module()

    with pytest.raises(ValueError, match="30 seconds or shorter"):
        audio.decode_wav(_wav(31))


def test_persisted_training_wav_accepts_and_caps_legacy_long_sample() -> None:
    audio = _load_audio_module()

    pcm, sample_rate = audio.decode_persisted_training_wav(_wav(31))

    assert sample_rate == 8_000
    assert len(pcm) == 30 * 8_000 * 2


def test_live_wav_can_exceed_upload_limit() -> None:
    audio = _load_audio_module()
    wav_data = _wav(31)

    pcm, sample_rate = audio.prepare_live_pcm(wav_data, 8_000, 1)

    assert sample_rate == 8_000
    assert len(pcm) == 31 * 8_000 * 2


def test_profile_rebuild_uses_persisted_training_decoder() -> None:
    recognition = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "recognition.py"
    ).read_text(encoding="utf-8")

    assert "decode_persisted_training_wav" in recognition
    assert "decode_persisted_training_wav, audio_data" in recognition
