"""Tests for evidence-based Home Assistant confidence calibration."""

from __future__ import annotations

from pathlib import Path
import runpy

ROOT = Path(__file__).parents[1] / "custom_components" / "speaker_recognition"
CALIBRATION = runpy.run_path(str(ROOT / "calibration.py"))
analyze_thresholds = CALIBRATION["analyze_thresholds"]
simulate_threshold = CALIBRATION["simulate_threshold"]


def _record(
    confidence: float,
    feedback: str,
    *,
    candidate: str = "user-a",
    actual: str | None = None,
    accepted: bool = True,
    identity_eligible: bool = True,
) -> dict[str, object]:
    return {
        "confidence": confidence,
        "accepted": accepted,
        "user_id": candidate if accepted else None,
        "candidate_user_id": candidate,
        "identity_eligible": identity_eligible,
        "feedback": feedback,
        "actual_user_id": actual,
    }


def test_recommendation_waits_for_enough_labelled_evidence() -> None:
    records = [_record(0.75, "correct") for _ in range(14)]

    result = analyze_thresholds(records, 0.70)

    assert result["ready"] is False
    assert result["minimum_labelled"] == 15
    assert result["recommended_threshold"] is None


def test_recommendation_can_lower_threshold_to_recover_confirmed_speaker() -> None:
    records = [
        _record(0.74, "missed_speaker", actual="user-a", identity_eligible=False)
        for _ in range(15)
    ]

    result = analyze_thresholds(records, 0.80)

    assert result["ready"] is True
    assert result["recommended_threshold"] == 0.74
    assert result["current_metrics"]["missed_speakers"] == 15
    assert result["recommended_metrics"]["correct_identity"] == 15


def test_false_accepts_are_weighted_more_heavily_than_misses() -> None:
    records = [
        *[
            _record(0.71, "correct", candidate="user-a", identity_eligible=True)
            for _ in range(10)
        ],
        *[
            _record(
                0.72,
                "wrong_speaker",
                candidate="user-b",
                actual="user-a",
                identity_eligible=True,
            )
            for _ in range(5)
        ],
    ]

    result = analyze_thresholds(records, 0.70)

    assert result["recommended_threshold"] == 0.73
    assert result["recommended_metrics"]["false_accepts"] == 0
    assert result["recommended_metrics"]["missed_speakers"] == 15


def test_backend_rejected_miss_is_not_claimed_as_threshold_fixable() -> None:
    records = [
        _record(
            0.50,
            "missed_speaker",
            actual="user-a",
            accepted=False,
            identity_eligible=False,
        )
        for _ in range(15)
    ]

    result = analyze_thresholds(records, 0.70)

    assert result["backend_rejected_misses"] == 15
    assert result["threshold_actionable_misses"] == 0
    assert result["recommended_metrics"]["missed_speakers"] == 15


def test_equal_best_score_keeps_current_threshold_when_possible() -> None:
    records = [_record(0.80, "correct") for _ in range(15)]

    result = analyze_thresholds(records, 0.70)

    assert result["recommended_threshold"] == 0.70
    assert simulate_threshold(records, 0.70).score == 0


def test_rejecting_unenrolled_voice_is_correct_rejection() -> None:
    """A threshold must not be penalised for rejecting a stranger."""
    records = [
        _record(0.72, "wrong_speaker", candidate="user-a", actual=None)
        for _ in range(15)
    ]
    accepted = simulate_threshold(records, 0.70)
    rejected = simulate_threshold(records, 0.73)
    assert accepted.false_accepts == 15
    assert accepted.correct_rejection == 0
    assert rejected.correct_rejection == 15
    assert rejected.missed_speakers == 0
    assert rejected.score == 0
    analysis = analyze_thresholds(records, 0.70)
    assert analysis["recommended_threshold"] == 0.73
    assert analysis["recommended_metrics"]["correct_rejection"] == 15


def test_rejecting_known_wrong_speaker_still_counts_as_a_miss() -> None:
    records = [
        _record(0.72, "wrong_speaker", candidate="user-a", actual="user-b")
    ]
    result = simulate_threshold(records, 0.73)
    assert result.missed_speakers == 1
    assert result.correct_rejection == 0


def test_unavailable_recognition_does_not_qualify_for_calibration() -> None:
    unavailable = {
        **_record(0.0, "correct", accepted=False, identity_eligible=False),
        "candidate_user_id": "",
        "recognition_status": "unavailable",
    }
    evaluated = _record(0.81, "correct")
    analysis = analyze_thresholds([unavailable] * 20 + [evaluated] * 14, 0.70)
    assert analysis["labelled_count"] == 14
    assert analysis["ready"] is False


def test_status_flag_excludes_invalid_trials_even_with_candidate() -> None:
    failed = {
        **_record(0.80, "correct"),
        "recognition_status": "unavailable",
    }
    analysis = analyze_thresholds([failed] * 15, 0.70)
    assert analysis["labelled_count"] == 0
    assert analysis["recommended_threshold"] is None
