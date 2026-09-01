"""Tests for embedding-engine abstraction and evaluation helpers."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import speaker_recognition.engines as engine_module
from speaker_recognition.engines import EngineInfo
from speaker_recognition.evaluation import (
    EvaluationRecord,
    compare_engines,
    decision_from_scores,
    evaluate_records,
    find_best_operating_point,
)
from speaker_recognition.models import AudioInput, Config, RecognitionRequest


class _FixedEngine:
    """Small deterministic engine used to exercise the generic recognizer."""

    def __init__(
        self,
        embedding: list[float],
        *,
        engine_id: str = "fake",
        display_name: str = "Fake Engine",
    ) -> None:
        self.info = EngineInfo(engine_id=engine_id, display_name=display_name)
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def prepare_audio(self, audio_input: AudioInput) -> np.ndarray:
        del audio_input
        return np.array([0.1, 0.2], dtype=np.float32)

    def embed_prepared(self, waveform: np.ndarray) -> np.ndarray:
        del waveform
        return self.embedding


@pytest.fixture
def recognizer_module(monkeypatch: pytest.MonkeyPatch):
    """Load recognizer with a cheap default engine for its module singleton."""
    monkeypatch.setattr(
        engine_module,
        "create_engine",
        lambda engine_id="resemblyzer": _FixedEngine([1.0, 0.0]),
    )
    module_path = Path(__file__).parents[1] / "speaker_recognition" / "recognizer.py"
    spec = importlib.util.spec_from_file_location("engine_eval_recognizer", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _audio_input() -> AudioInput:
    pcm = (1000).to_bytes(2, "little", signed=True) * 100
    return AudioInput(audio_data=base64.b64encode(pcm).decode(), sample_rate=16000)


def test_recognizer_scores_through_injected_engine_before_acceptance(
    recognizer_module, tmp_path: Path
) -> None:
    """Raw scores remain available even when the normal policy rejects them."""
    recognizer = recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(tmp_path / "embeddings")),
        engine=_FixedEngine([1.0, 0.0]),
    )
    alice = np.array([0.50, np.sqrt(0.75)], dtype=np.float32)
    recognizer._reference_embeddings = {"alice": alice}
    recognizer._sample_embeddings = {"alice": alice.reshape(1, -1)}
    recognizer._is_trained = True

    scores = recognizer.score(RecognitionRequest(audio=_audio_input()))
    result = recognizer.recognize(RecognitionRequest(audio=_audio_input()))

    assert scores.engine_id == "fake"
    assert scores.candidate_user_id == "alice"
    assert scores.similarity == 0.5
    assert result.candidate_user_id == "alice"
    assert result.accepted is False
    assert result.user_id is None


def test_profile_engine_metadata_prevents_cross_engine_loading(
    recognizer_module, tmp_path: Path
) -> None:
    """A profile written by one engine is ignored by a different engine."""
    directory = tmp_path / "embeddings"
    first = recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(directory)),
        engine=_FixedEngine([1.0, 0.0], engine_id="engine-a"),
    )
    first._persist_profiles_transactionally(
        {
            "alice": (
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([[1.0, 0.0]], dtype=np.float32),
            )
        }
    )

    same_engine = recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(directory)),
        engine=_FixedEngine([1.0, 0.0], engine_id="engine-a"),
    )
    other_engine = recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(directory)),
        engine=_FixedEngine([1.0, 0.0], engine_id="engine-b"),
    )

    assert same_engine.enrolled_users == ["alice"]
    assert other_engine.enrolled_users == []


def test_legacy_v1_profiles_remain_usable_for_resemblyzer(
    recognizer_module, tmp_path: Path
) -> None:
    """Existing deployments keep their pre-engine-metadata profiles."""
    directory = tmp_path / "embeddings"
    directory.mkdir()
    digest = hashlib.sha256(b"alice").hexdigest()
    np.savez(
        directory / f"{digest}_profile.npz",
        schema_version=np.array(1, dtype=np.int16),
        user_id=np.array("alice"),
        centroid=np.array([1.0, 0.0], dtype=np.float32),
        sample_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
    )

    recognizer = recognizer_module.SpeakerRecognizer(
        Config(embeddings_directory=str(directory)),
        engine=_FixedEngine([1.0, 0.0], engine_id="resemblyzer"),
    )

    assert recognizer.enrolled_users == ["alice"]


def test_open_set_evaluation_distinguishes_error_types() -> None:
    """Evaluation separates misses, wrong speakers, and unknown false accepts."""
    records = [
        EvaluationRecord("engine-a", {"alice": 0.80, "bob": 0.30}, "alice"),
        EvaluationRecord("engine-a", {"alice": 0.52, "bob": 0.30}, "alice"),
        EvaluationRecord("engine-a", {"alice": 0.40, "bob": 0.75}, "alice"),
        EvaluationRecord("engine-a", {"alice": 0.62, "bob": 0.40}, None),
        EvaluationRecord("engine-a", {"alice": 0.30, "bob": 0.25}, None),
    ]

    metrics = evaluate_records(
        records, similarity_threshold=0.55, margin_threshold=0.05
    )

    assert metrics.correct_known == 1
    assert metrics.false_unknowns == 1
    assert metrics.wrong_speaker == 1
    assert metrics.false_accepts == 1
    assert metrics.correct_unknown == 1
    assert metrics.false_identifications == 2
    assert metrics.trials == 5


def test_engine_comparison_uses_identical_policy_per_engine() -> None:
    """Shadow engines can be compared from the same labelled trial format."""
    records = [
        EvaluationRecord("resemblyzer", {"alice": 0.70, "bob": 0.40}, "alice"),
        EvaluationRecord("ecapa", {"alice": 0.82, "bob": 0.20}, "alice"),
        EvaluationRecord("resemblyzer", {"alice": 0.60, "bob": 0.58}, None),
        EvaluationRecord("ecapa", {"alice": 0.40, "bob": 0.35}, None),
    ]

    compared = compare_engines(
        records, similarity_threshold=0.55, margin_threshold=0.05
    )

    assert compared["resemblyzer"].correct == 2
    assert compared["ecapa"].correct == 2
    assert set(compared) == {"ecapa", "resemblyzer"}


def test_threshold_search_prefers_avoiding_false_identification() -> None:
    """The default cost makes a false identity costlier than an unknown result."""
    records = [
        EvaluationRecord("engine-a", {"alice": 0.80, "bob": 0.20}, "alice"),
        EvaluationRecord("engine-a", {"alice": 0.58, "bob": 0.20}, None),
    ]

    best = find_best_operating_point(
        records,
        similarity_thresholds=[0.55, 0.60],
        margin_thresholds=[0.0],
    )

    assert best.similarity_threshold == 0.60
    assert best.metrics.false_accepts == 0
    assert best.metrics.correct == 2


def test_decision_recomputes_margin_from_raw_scores() -> None:
    """Threshold experiments do not rely on an engine's original accepted flag."""
    scores = {"alice": 0.70, "bob": 0.68}
    assert decision_from_scores(scores, 0.55, 0.01) == "alice"
    assert decision_from_scores(scores, 0.55, 0.05) is None
