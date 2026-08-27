"""Focused tests for recognition scoring and unknown-speaker decisions."""

import base64
import importlib.util
from pathlib import Path

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


class _DummyEncoder:
    """Return a configurable deterministic embedding."""

    def __init__(self) -> None:
        self.embedding = np.array([1.0, 0.0], dtype=np.float32)

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return self.embedding


class _SequenceEncoder:
    """Return predetermined embeddings in order."""

    def __init__(self, embeddings: list[list[float]]) -> None:
        self._embeddings = iter(embeddings)

    def embed_utterance(self, wav: np.ndarray) -> np.ndarray:
        del wav
        return np.asarray(next(self._embeddings), dtype=np.float32)


def _audio_input() -> AudioInput:
    pcm = (1000).to_bytes(2, "little", signed=True) * 200
    return AudioInput(audio_data=base64.b64encode(pcm).decode(), sample_rate=16000)


@pytest.fixture
def recognizer_module(monkeypatch: pytest.MonkeyPatch):
    """Load recognizer without initializing the real Torch encoder."""
    monkeypatch.setattr(resemblyzer, "VoiceEncoder", _DummyEncoder)
    module_path = Path(__file__).parents[1] / "speaker_recognition" / "recognizer.py"
    spec = importlib.util.spec_from_file_location("decision_recognizer_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recognizer(recognizer_module, tmp_path: Path):
    return recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(tmp_path / "embeddings"))
    )


def test_individual_enrollment_samples_influence_profile_score(
    recognizer_module, tmp_path: Path
) -> None:
    """A strong sample-level match contributes alongside the centroid."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    chunk = np.array([1.0, 0.0], dtype=np.float32)
    centroid = recognizer._normalize_embedding(np.array([1.0, 1.0]))
    samples = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    score = recognizer._profile_score(centroid, samples, chunk)

    assert score > float(np.dot(centroid, chunk))
    assert score == pytest.approx((1 / np.sqrt(2) + 1.0) / 2)


def test_ambiguous_best_match_is_rejected_by_margin(
    recognizer_module, tmp_path: Path
) -> None:
    """Two nearly equal candidates produce an explicit unknown decision."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    alice = np.array([0.80, 0.60], dtype=np.float32)
    bob = np.array([0.78, np.sqrt(1.0 - 0.78**2)], dtype=np.float32)
    recognizer._reference_embeddings = {"alice": alice, "bob": bob}
    recognizer._sample_embeddings = {
        "alice": alice.reshape(1, -1),
        "bob": bob.reshape(1, -1),
    }
    recognizer._is_trained = True

    result = recognizer.recognize(RecognitionRequest(audio=_audio_input()))

    assert result.candidate_user_id == "alice"
    assert result.margin == pytest.approx(0.02, abs=1e-6)
    assert not result.accepted
    assert result.user_id is None


def test_low_similarity_is_rejected_even_with_one_enrolled_user(
    recognizer_module, tmp_path: Path
) -> None:
    """Nearest-neighbour matching does not force a weak speaker identity."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    alice = np.array([0.50, np.sqrt(0.75)], dtype=np.float32)
    recognizer._reference_embeddings = {"alice": alice}
    recognizer._sample_embeddings = {"alice": alice.reshape(1, -1)}
    recognizer._is_trained = True

    result = recognizer.recognize(RecognitionRequest(audio=_audio_input()))

    assert result.candidate_user_id == "alice"
    assert result.margin is None
    assert result.similarity == pytest.approx(0.50)
    assert not result.accepted
    assert result.user_id is None


def test_strong_single_user_match_is_accepted(
    recognizer_module, tmp_path: Path
) -> None:
    """Margin is not required when there is no runner-up speaker."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    alice = np.array([0.90, np.sqrt(0.19)], dtype=np.float32)
    recognizer._reference_embeddings = {"alice": alice}
    recognizer._sample_embeddings = {"alice": alice.reshape(1, -1)}
    recognizer._is_trained = True

    result = recognizer.recognize(RecognitionRequest(audio=_audio_input()))

    assert result.accepted
    assert result.user_id == "alice"
    assert result.candidate_user_id == "alice"
    assert result.confidence == result.similarity


def test_enrollment_reports_consistency_and_outlier_sample(
    recognizer_module, tmp_path: Path
) -> None:
    """Training reports a recording that disagrees strongly with its peers."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    recognizer._encoder = _SequenceEncoder(
        [[1.0, 0.0], [0.99, 0.10], [0.98, 0.20], [-1.0, 0.0]]
    )
    sample = _audio_input()

    result = recognizer.train(
        TrainingRequest(
            voice_samples=[VoiceSample(user="alice", audio=sample) for _ in range(4)]
        )
    )

    assert "alice" in result.profile_consistency
    assert -1.0 <= result.profile_consistency["alice"] <= 1.0
    assert result.outlier_samples["alice"] == [4]


def test_persisted_sample_embeddings_are_loaded_for_matching(
    recognizer_module, tmp_path: Path
) -> None:
    """Restarted recognizers retain sample-level references, not only centroids."""
    recognizer = _recognizer(recognizer_module, tmp_path)
    recognizer._encoder = _SequenceEncoder([[1.0, 0.0], [0.0, 1.0]])
    sample = _audio_input()
    recognizer.train(
        TrainingRequest(
            voice_samples=[
                VoiceSample(user="alice", audio=sample),
                VoiceSample(user="alice", audio=sample),
            ]
        )
    )

    restarted = _recognizer(recognizer_module, tmp_path)

    assert restarted._sample_embeddings["alice"].shape == (2, 2)
    np.testing.assert_allclose(
        np.linalg.norm(restarted._sample_embeddings["alice"], axis=1), [1.0, 1.0]
    )
