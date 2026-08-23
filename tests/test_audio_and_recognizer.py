"""Focused regression tests for audio conversion and embedding persistence."""

import base64
import importlib.util
import logging
from io import BytesIO
from pathlib import Path
import wave

import numpy as np
import pytest
import resemblyzer

from speaker_recognition.models import AudioInput, Config, RecognitionRequest, TrainingRequest, VoiceSample


def _load_integration_audio_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "audio.py"
    )
    spec = importlib.util.spec_from_file_location("speaker_recognition_integration_audio", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav_bytes(frames: bytes, *, sample_rate: int = 16000, channels: int = 1) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return buffer.getvalue()


def _audio_input(samples: list[int], sample_rate: int = 16000) -> AudioInput:
    pcm = b"".join(sample.to_bytes(2, "little", signed=True) for sample in samples)
    return AudioInput(
        audio_data=base64.b64encode(pcm).decode(), sample_rate=sample_rate
    )


def test_wav_training_audio_is_decoded_to_mono_pcm() -> None:
    """Training WAV containers are decoded instead of sent as PCM bytes."""
    audio = _load_integration_audio_module()
    wav_data = _wav_bytes(
        b"".join(
            left.to_bytes(2, "little", signed=True)
            + right.to_bytes(2, "little", signed=True)
            for left, right in [(1000, -1000), (3000, 1000)]
        ),
        sample_rate=22050,
        channels=2,
    )

    pcm, sample_rate = audio.decode_wav(wav_data)

    assert sample_rate == 22050
    assert pcm == b"\x00\x00\xd0\x07"


def test_raw_pcm_live_audio_is_downmixed_using_metadata() -> None:
    """Live PCM frames do not need a WAV container to be recognized."""
    audio = _load_integration_audio_module()
    raw_stereo_pcm = b"\xe8\x03\x18\xfc\xb8\x0b\xe8\x03"

    pcm, sample_rate = audio.prepare_live_pcm(raw_stereo_pcm, 22050, 2)

    assert sample_rate == 22050
    assert pcm == b"\x00\x00\xd0\x07"


def test_wav_container_live_audio_is_decoded() -> None:
    """A live WAV container is decoded when its header is present."""
    audio = _load_integration_audio_module()
    wav_data = _wav_bytes(b"\xe8\x03\x18\xfc", sample_rate=16000, channels=2)

    pcm, sample_rate = audio.prepare_live_pcm(wav_data, 22050, 2)

    assert sample_rate == 16000
    assert pcm == b"\x00\x00"


def test_non_wav_training_audio_is_rejected() -> None:
    """MP3 is not advertised because no reliable decoder is bundled with HA."""
    audio = _load_integration_audio_module()

    with pytest.raises(ValueError, match="Only uncompressed 16-bit PCM WAV"):
        audio.decode_wav(b"ID3 not a wav file")


class _DummyEncoder:
    """Fast deterministic encoder for recognizer persistence tests."""

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return np.array([1.0, 0.0], dtype=np.float32)


@pytest.fixture
def recognizer_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Load the recognizer with a dummy encoder without initializing Torch globally."""
    monkeypatch.setattr(resemblyzer, "VoiceEncoder", _DummyEncoder)
    module_path = Path(__file__).parents[1] / "speaker_recognition" / "recognizer.py"
    spec = importlib.util.spec_from_file_location("test_recognizer_module", module_path)
    assert spec and spec.loader
    recognizer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recognizer_module)

    def factory():
        return recognizer_module.SpeakerRecognizer(
            Config(embeddings_directory=str(tmp_path / "embeddings"))
        )

    return factory


def test_train_then_recognize_and_load_persisted_embeddings(recognizer_factory) -> None:
    """Saved embeddings make a newly started backend immediately usable."""
    recognizer = recognizer_factory()
    sample = _audio_input([0, 1000, -1000, 500])
    training = recognizer.train(
        TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=sample)])
    )

    assert training.trained_users == ["alice"]
    assert recognizer.recognize(RecognitionRequest(audio=sample)).user_id == "alice"

    restarted_recognizer = recognizer_factory()
    assert restarted_recognizer.is_trained
    assert restarted_recognizer.recognize(RecognitionRequest(audio=sample)).user_id == "alice"


def test_failed_training_logs_error_and_leaves_model_untrained(
    recognizer_factory, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed sample is visible and cannot leave stale trained state behind."""
    recognizer = recognizer_factory()
    invalid = AudioInput(audio_data=base64.b64encode(b"\x00").decode(), sample_rate=16000)

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="No valid"):
        recognizer.train(
            TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=invalid)])
        )

    assert not recognizer.is_trained
    assert "Error processing voice sample for user alice" in caplog.text
