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


class _SequenceEncoder:
    """Return predetermined embeddings for aggregation tests."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = iter(embeddings)

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return np.asarray(next(self._embeddings), dtype=np.float32)


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
    assert restarted_recognizer.enrolled_users == ["alice"]
    assert (
        restarted_recognizer.recognize(RecognitionRequest(audio=sample)).user_id
        == "alice"
    )


def test_legacy_single_embedding_is_loaded(recognizer_factory) -> None:
    """Version 1 reference files remain usable until the user is retrained."""
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
    """Every sample remains available while recognition uses a normalized mean."""
    recognizer = recognizer_factory()
    recognizer._encoder = _SequenceEncoder([[3.0, 0.0], [0.0, 1.0]])
    first = _audio_input([100, 200])
    second = _audio_input([300, 400])

    result = recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(user="alice", audio=first),
                VoiceSample(user="alice", audio=second),
            ]
        )
    )

    assert result.accepted_samples == {"alice": 2}
    profile_path = next(recognizer.embeddings_directory.glob("*_profile.npz"))
    with np.load(profile_path, allow_pickle=False) as profile:
        np.testing.assert_allclose(
            profile["sample_embeddings"], [[3.0, 0.0], [0.0, 1.0]]
        )
        np.testing.assert_allclose(
            profile["centroid"], np.array([3.0, 1.0]) / np.sqrt(10.0)
        )
        assert np.linalg.norm(profile["centroid"]) == pytest.approx(1.0)


def test_retraining_one_user_preserves_unrelated_profiles(recognizer_factory) -> None:
    """The training API upserts requested users rather than clearing the model."""
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0], [0.0, 1.0]])
    recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(user="alice", audio=sample),
                VoiceSample(user="bob", audio=sample),
            ]
        )
    )

    recognizer._encoder = _SequenceEncoder(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    )
    result = recognizer.train(
        TrainingRequest(
            voice_samples=[VoiceSample(user="alice", audio=sample) for _ in range(3)]
        )
    )

    assert result.trained_users == ["alice"]
    assert result.count == 2
    restarted = recognizer_factory()
    assert set(restarted._reference_embeddings) == {"alice", "bob"}
    np.testing.assert_allclose(restarted._reference_embeddings["bob"], [0.0, 1.0])

    invalid = AudioInput(
        audio_data=base64.b64encode(b"\x00").decode(), sample_rate=16000
    )
    with pytest.raises(ValueError, match="No valid"):
        recognizer.train(
            TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=invalid)])
        )
    recognizer._encoder = _SequenceEncoder([[0.0, 1.0]])
    assert recognizer.recognize(RecognitionRequest(audio=sample)).user_id == "bob"


def test_retraining_rejects_too_few_accepted_samples(recognizer_factory) -> None:
    """A weak retraining attempt cannot replace an existing usable profile."""
    recognizer = recognizer_factory()
    sample = _audio_input([100, 200])
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0]])
    recognizer.train(
        TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=sample)])
    )
    original = recognizer._reference_embeddings["alice"].copy()

    recognizer._encoder = _SequenceEncoder([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="No valid"):
        recognizer.train(
            TrainingRequest(
                voice_samples=[VoiceSample(user="alice", audio=sample) for _ in range(2)]
            )
        )

    np.testing.assert_allclose(recognizer._reference_embeddings["alice"], original)


def test_invalid_sample_is_rejected_without_discarding_valid_sample(
    recognizer_factory,
) -> None:
    """An invalid recording is counted and skipped within a multi-sample profile."""
    recognizer = recognizer_factory()
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0]])
    invalid = _audio_input([0, 0, 0, 0])
    valid = _audio_input([100, 200])

    result = recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(user="alice", audio=invalid),
                VoiceSample(user="alice", audio=valid),
            ]
        )
    )

    assert result.accepted_samples == {"alice": 1}
    assert result.rejected_samples == {"alice": 1}
    assert recognizer.is_trained


def test_failed_training_logs_error_and_leaves_model_untrained(
    recognizer_factory, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed sample is visible and cannot leave stale trained state behind."""
    recognizer = recognizer_factory()
    invalid = AudioInput(
        audio_data=base64.b64encode(b"\x00").decode(), sample_rate=16000
    )

    with caplog.at_level(logging.ERROR), pytest.raises(ValueError, match="No valid"):
        recognizer.train(
            TrainingRequest(voice_samples=[VoiceSample(user="alice", audio=invalid)])
        )

    assert not recognizer.is_trained
    assert "Error processing voice sample 1 for user alice" in caplog.text
