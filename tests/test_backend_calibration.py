"""Joint calibration uses explicit labels and persisted raw evidence only."""

from pathlib import Path
import runpy

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "custom_components/speaker_recognition/calibration.py"))
analyze = MODULE["analyze_backend_thresholds"]
DEFAULT = {"min_similarity": 0.55, "min_margin": 0.05}


def record(similarity=0.8, margin=0.1, *, actual="alice", candidate="alice", accepted=True):
    return dict(similarity=similarity, margin=margin, candidate_user_id=candidate,
                actual_user_id=actual, feedback="wrong_speaker" if actual != candidate else "missed_speaker",
                accepted=accepted, user_id=candidate if accepted else None)


def test_insufficient_and_invalid_evidence():
    assert not analyze([record()] * 14, DEFAULT)["ready"]
    invalid = [dict(record(), similarity=float("nan")), dict(record(), margin=float("inf")),
               {"feedback": "correct", "confidence": 0.9}, dict(record(), candidate_user_id="")]
    assert not analyze([record()] * 14 + invalid, DEFAULT)["ready"]
    assert analyze(invalid, DEFAULT)["recommended_policy"] is None


def test_similarity_sensitive_misses_recovered():
    result = analyze([record(0.51, accepted=False)] * 15, DEFAULT)
    assert result["current_metrics"]["false_unknowns"] == 15
    assert result["recommended_policy"]["min_similarity"] == 0.5
    assert result["recommended_metrics"]["correct_identity"] == 15


def test_margin_sensitive_misses_recovered():
    result = analyze([record(margin=0.02, accepted=False)] * 15, DEFAULT)
    assert result["recommended_policy"] == {"min_similarity": 0.55, "min_margin": 0.0}
    assert result["recommended_metrics"]["correct_identity"] == 15


def test_false_accepts_outweigh_misses():
    rows = [record()] * 10 + [record(actual="bob")] * 5
    result = analyze(rows, DEFAULT)
    assert result["recommended_metrics"]["wrong_speaker"] == 0
    assert result["recommended_metrics"]["false_unknowns"] == 15
    assert result["recommended_metrics"]["score"] < result["current_metrics"]["score"]


def test_unknown_examples_and_margin_discrimination():
    rows = [record(margin=0.3)] * 10 + [record(margin=0.06, actual=None)] * 5
    result = analyze(rows, DEFAULT)
    assert result["recommended_policy"] == {"min_similarity": 0.55, "min_margin": 0.1}
    assert result["recommended_metrics"]["correct_rejection"] == 5
    assert result["recommended_metrics"]["correct_identity"] == 10


def test_legacy_correct_feedback_and_single_profile():
    known = dict(record(margin=None), feedback="correct", identity_eligible=True, user_id="alice", actual_user_id=None)
    unknown = dict(record(0.4, margin=None), feedback="correct", identity_eligible=False, user_id=None)
    result = analyze([known] * 10 + [unknown] * 5, DEFAULT)
    assert result["recommended_policy"] == DEFAULT
    assert result["current_metrics"]["correct_identity"] == 10
    assert result["current_metrics"]["correct_rejection"] == 5


def test_deterministic_and_off_grid_policy_preserved():
    policy = {"min_similarity": 0.553, "min_margin": 0.05}
    rows = [record(0.551, actual=None, margin=None)] * 5 + [record(0.8, margin=0.06, actual=None)] * 5 + [record()] * 5
    result = analyze(rows, policy)
    assert result == analyze(list(reversed(rows)), policy)
    recommended = result["recommended_policy"]
    # Re-evaluation at the actual recommendation must agree with advertised metrics.
    checked = analyze(rows, recommended)
    assert checked["current_metrics"] == result["recommended_metrics"]
    assert analyze([record()] * 15, policy)["recommended_policy"] == policy
