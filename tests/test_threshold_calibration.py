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
