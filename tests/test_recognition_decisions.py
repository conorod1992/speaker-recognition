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


def test_enrollment_reports_and_excludes_outlier_sample(
    recognizer_module, tmp_path: Path
) -> None:
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
    assert result.accepted_samples["alice"] == 3
    assert result.rejected_samples["alice"] == 1
    assert recognizer._sample_embeddings["alice"].shape == (3, 2)
    assert np.all(recognizer._sample_embeddings["alice"][:, 0] > 0)


def test_persisted_sample_embeddings_are_loaded_for_matching(
    recognizer_module, tmp_path: Path
) -> None:
    recognizer = _recognizer(recognizer_module, tmp_path)
    recognizer._encoder = _SequenceEncoder(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
    )
    sample = _audio_input()
    recognizer.train(
        TrainingRequest(
            voice_samples=[VoiceSample(user="alice", audio=sample) for _ in range(3)]
        )
    )
    restarted = _recognizer(recognizer_module, tmp_path)
    assert restarted._sample_embeddings["alice"].shape == (3, 2)
    np.testing.assert_allclose(
        np.linalg.norm(restarted._sample_embeddings["alice"], axis=1),
        [1.0, 1.0, 1.0],
    )


@pytest.mark.parametrize(
    "similarity,margin,min_similarity,min_margin,accepted",
    [
        (0.549999, 0.1, 0.55, 0.05, False),
        (0.55, 0.05, 0.55, 0.05, True),
        (0.550001, 0.1, 0.55, 0.05, True),
        (0.8, 0.049999, 0.55, 0.05, False),
        (0.8, 0.050001, 0.55, 0.05, True),
        (0.749999, 0.2, 0.75, 0.05, False),
        (0.750001, 0.2, 0.75, 0.05, True),
        (0.8, 0.149999, 0.55, 0.15, False),
        (0.8, 0.150001, 0.55, 0.15, True),
        (0.8, 0.0, 0.55, 0.0, True),
        (0.54, 0.0, 0.55, 0.0, False),
        (0.8, None, 0.55, 1.0, True),
    ],
)
def test_configured_acceptance_boundaries(
    recognizer_module, tmp_path, monkeypatch,
    similarity, margin, min_similarity, min_margin, accepted,
):
    from speaker_recognition.models import AcceptanceThresholds, RecognitionScores

    recognizer = _recognizer(recognizer_module, tmp_path)
    monkeypatch.setattr(recognizer, "score", lambda request: RecognitionScores(
        candidate_user_id="alice", similarity=similarity, margin=margin,
        all_scores={"alice": similarity},
    ))
    policy = AcceptanceThresholds(min_similarity=min_similarity, min_margin=min_margin)
    result = recognizer.recognize(RecognitionRequest(
        audio=_audio_input(), acceptance_thresholds=policy,
    ))
    assert result.accepted is accepted
    assert result.acceptance_thresholds == policy
    assert result.user_id == ("alice" if accepted else None)
    # Request policy is isolated: it never changes defaults for other callers.
    assert recognizer._config.acceptance_thresholds == AcceptanceThresholds()


def test_default_acceptance_policy_and_restart(recognizer_module, tmp_path):
    from speaker_recognition.models import AcceptanceThresholds

    policy = AcceptanceThresholds(min_similarity=0.7, min_margin=0.0)
    config = Config(embeddings_directory=str(tmp_path), acceptance_thresholds=policy)
    restored = Config.model_validate_json(config.model_dump_json())
    assert recognizer_module.SpeakerRecognizer(restored)._config.acceptance_thresholds == policy
    assert Config().acceptance_thresholds.model_dump() == {
        "min_similarity": 0.55, "min_margin": 0.05,
    }


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["min_similarity", "min_margin"])
def test_invalid_acceptance_policy(value, field):
    from pydantic import ValidationError
    from speaker_recognition.models import AcceptanceThresholds

    with pytest.raises(ValidationError):
        AcceptanceThresholds(**{field: value})
