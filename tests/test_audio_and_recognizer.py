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

from speaker_recognition.models import (
    AudioInput,
    Config,
    RecognitionRequest,
    TrainingRequest,
    VoiceSample,
)


def _load_integration_audio_module():
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


def _samples(user: str, audio: AudioInput, count: int = 3) -> list[VoiceSample]:
    return [VoiceSample(user=user, audio=audio) for _ in range(count)]


def test_wav_training_audio_is_decoded_to_mono_pcm() -> None:
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
    audio = _load_integration_audio_module()
    raw_stereo_pcm = b"\xe8\x03\x18\xfc\xb8\x0b\xe8\x03"
    pcm, sample_rate = audio.prepare_live_pcm(raw_stereo_pcm, 22050, 2)
    assert sample_rate == 22050
    assert pcm == b"\x00\x00\xd0\x07"


def test_wav_container_live_audio_is_decoded() -> None:
    audio = _load_integration_audio_module()
    wav_data = _wav_bytes(b"\xe8\x03\x18\xfc", sample_rate=16000, channels=2)
    pcm, sample_rate = audio.prepare_live_pcm(wav_data, 22050, 2)
    assert sample_rate == 16000
    assert pcm == b"\x00\x00"


def test_non_wav_training_audio_is_rejected() -> None:
    audio = _load_integration_audio_module()
    with pytest.raises(ValueError, match="Only uncompressed 16-bit PCM WAV"):
        audio.decode_wav(b"ID3 not a wav file")


class _DummyEncoder:
    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return np.array([1.0, 0.0], dtype=np.float32)


class _SequenceEncoder:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = iter(embeddings)

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return np.asarray(next(self._embeddings), dtype=np.float32)


@pytest.fixture
def recognizer_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
    recognizer = recognizer_factory()
    sample = _audio_input([0, 1000, -1000, 500])
    training = recognizer.train(TrainingRequest(voice_samples=_samples("alice", sample)))
    assert training.trained_users == ["alice"]
    assert training.accepted_samples == {"alice": 3}
    assert recognizer.recognize(RecognitionRequest(audio=sample)).user_id == "alice"

    restarted_recognizer = recognizer_factory()
    assert restarted_recognizer.is_trained
    assert restarted_recognizer.enrolled_users == ["alice"]
    assert restarted_recognizer.recognize(RecognitionRequest(audio=sample)).user_id == "alice"


def test_legacy_single_embedding_is_loaded(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    recognizer.embeddings_directory.mkdir(parents=True)
    np.save(
        recognizer.embeddings_directory / "legacy_embedding.npy",
        np.array([3.0, 4.0], dtype=np.float32),
    )
    restarted = recognizer_factory()
    assert restarted.is_trained
    np.testing.assert_allclose(restarted._reference_embeddings["legacy"], [0.6, 0.8])


def test_multiple_samples_are_averaged_and_centroid_is_normalized(
    recognizer_factory,
) -> None:
    recognizer = recognizer_factory()
    recognizer._encoder = _SequenceEncoder(
        [[3.0, 0.0], [3.0, 0.0], [0.0, 1.0]]
    )
    sample = _audio_input([100, 200])
    result = recognizer.train(
        TrainingRequest(voice_samples=_samples("alice", sample))
    )
    assert result.accepted_samples == {"alice": 3}
    profile_path = next(recognizer.embeddings_directory.glob("*_profile.npz"))
    with np.load(profile_path, allow_pickle=False) as profile:
        assert profile["sample_embeddings"].shape == (3, 2)
        assert np.linalg.norm(profile["centroid"]) == pytest.approx(1.0)


def test_retraining_one_user_preserves_unrelated_profiles(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3
    )
    recognizer.train(
        TrainingRequest(
            voice_samples=_samples("alice", sample) + _samples("bob", sample)
        )
    )

    recognizer._encoder = _SequenceEncoder([[1.0, 1.0]] * 3)
    result = recognizer.train(
        TrainingRequest(voice_samples=_samples("alice", sample))
    )
    assert result.trained_users == ["alice"]
    assert result.count == 2
    restarted = recognizer_factory()
    assert set(restarted._reference_embeddings) == {"alice", "bob"}
    np.testing.assert_allclose(restarted._reference_embeddings["bob"], [0.0, 1.0])


def test_new_enrollment_rejects_too_few_accepted_samples(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least 3"):
        recognizer.train(
            TrainingRequest(voice_samples=_samples("alice", sample, 2))
        )
    assert not recognizer.is_trained
    assert not list(recognizer.embeddings_directory.glob("*_profile.npz"))


def test_retraining_rejects_too_few_accepted_samples(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0]] * 3)
    recognizer.train(TrainingRequest(voice_samples=_samples("alice", sample)))
    original = recognizer._reference_embeddings["alice"].copy()

    recognizer._encoder = _SequenceEncoder([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="at least 3"):
        recognizer.train(
            TrainingRequest(voice_samples=_samples("alice", sample, 2))
        )
    np.testing.assert_allclose(recognizer._reference_embeddings["alice"], original)


def test_invalid_sample_requires_three_remaining_usable_samples(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0]] * 3)
    invalid = _audio_input([0, 0, 0, 0])
    valid = _audio_input([100, 200])
    result = recognizer.train(
        TrainingRequest(
            voice_samples=[VoiceSample(user="alice", audio=invalid)]
            + _samples("alice", valid)
        )
    )
    assert result.accepted_samples == {"alice": 3}
    assert result.rejected_samples == {"alice": 1}
    assert recognizer.is_trained


def test_multi_user_training_is_all_or_nothing(recognizer_factory) -> None:
    """A bad second user cannot leave the first user's profile partially updated."""
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0]] * 3)
    recognizer.train(TrainingRequest(voice_samples=_samples("alice", sample)))
    original = recognizer._reference_embeddings["alice"].copy()

    recognizer._encoder = _SequenceEncoder(
        [[0.8, 0.2]] * 3 + [[0.0, 1.0]] * 2
    )
    with pytest.raises(ValueError, match="bob.*at least 3"):
        recognizer.train(
            TrainingRequest(
                voice_samples=_samples("alice", sample) + _samples("bob", sample, 2)
            )
        )

    np.testing.assert_allclose(recognizer._reference_embeddings["alice"], original)
    assert "bob" not in recognizer._reference_embeddings
    restarted = recognizer_factory()
    np.testing.assert_allclose(restarted._reference_embeddings["alice"], original)
    assert "bob" not in restarted._reference_embeddings


def test_sync_profiles_removes_stale_profile_from_disk_and_memory(recognizer_factory) -> None:
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3
    )
    recognizer.train(
        TrainingRequest(
            voice_samples=_samples("alice", sample) + _samples("bob", sample)
        )
    )

    assert recognizer.sync_profiles({"alice"}) == ["bob"]
    assert recognizer.enrolled_users == ["alice"]
    restarted = recognizer_factory()
    assert restarted.enrolled_users == ["alice"]


def test_failed_training_logs_error_and_leaves_model_untrained(
    recognizer_factory, caplog: pytest.LogCaptureFixture
) -> None:
    recognizer = recognizer_factory()
    invalid = AudioInput(
        audio_data=base64.b64encode(b"\x00").decode(), sample_rate=16000
    )
    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="at least 3"):
        recognizer.train(
            TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=invalid)])
        )
    assert not recognizer.is_trained
    assert "Error processing voice sample 1 for user alice" in caplog.text
