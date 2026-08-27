"""Analyze labelled recognition history and recommend a conservative HA threshold."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

MIN_LABELLED_DECISIONS = 15
FALSE_ACCEPT_WEIGHT = 5
MISSED_SPEAKER_WEIGHT = 1
_THRESHOLD_STEP = 0.01


@dataclass(frozen=True)
class ThresholdMetrics:
    """Simulated outcomes for one Home Assistant confidence threshold."""

    threshold: float
    correct_identity: int
    correct_rejection: int
    false_accepts: int
    missed_speakers: int
    score: int


def _would_apply_identity(record: dict[str, Any], threshold: float) -> bool:
    """Return whether HA would apply the backend candidate at this threshold."""
    return bool(
        record.get("accepted")
        and record.get("user_id")
        and float(record.get("confidence", 0.0)) >= threshold
    )


def _classify(record: dict[str, Any], threshold: float) -> tuple[str, int]:
    """Classify one labelled historical decision under a simulated threshold."""
    feedback = record.get("feedback")
    applies = _would_apply_identity(record, threshold)
    candidate_user_id = record.get("user_id") or record.get("candidate_user_id")
    actual_user_id = record.get("actual_user_id")

    if feedback == "correct":
        originally_applied = bool(record.get("identity_eligible") and record.get("user_id"))
        if originally_applied:
            if applies:
                return "correct_identity", 0
            return "missed_speaker", MISSED_SPEAKER_WEIGHT
        if applies:
            return "false_accept", FALSE_ACCEPT_WEIGHT
        return "correct_rejection", 0

    if feedback == "wrong_speaker":
        if applies and candidate_user_id != actual_user_id:
            return "false_accept", FALSE_ACCEPT_WEIGHT
        if applies and candidate_user_id == actual_user_id:
            return "correct_identity", 0
        return "missed_speaker", MISSED_SPEAKER_WEIGHT

    if feedback == "missed_speaker":
        if applies and candidate_user_id == actual_user_id:
            return "correct_identity", 0
        if applies:
            return "false_accept", FALSE_ACCEPT_WEIGHT
        return "missed_speaker", MISSED_SPEAKER_WEIGHT

    raise ValueError("Decision is not labelled")


def simulate_threshold(
    records: Iterable[dict[str, Any]], threshold: float
) -> ThresholdMetrics:
    """Simulate labelled decisions at one HA confidence threshold."""
    counts = {
        "correct_identity": 0,
        "correct_rejection": 0,
        "false_accept": 0,
        "missed_speaker": 0,
    }
    score = 0
    for record in records:
        if record.get("feedback") not in ("correct", "wrong_speaker", "missed_speaker"):
            continue
        outcome, penalty = _classify(record, threshold)
        counts[outcome] += 1
        score += penalty

    return ThresholdMetrics(
        threshold=round(threshold, 2),
        correct_identity=counts["correct_identity"],
        correct_rejection=counts["correct_rejection"],
        false_accepts=counts["false_accept"],
        missed_speakers=counts["missed_speaker"],
        score=score,
    )


def analyze_thresholds(
    records: Iterable[dict[str, Any]], current_threshold: float
) -> dict[str, Any]:
    """Return conservative threshold guidance from explicitly labelled decisions."""
    labelled = [
        dict(record)
        for record in records
        if record.get("feedback") in ("correct", "wrong_speaker", "missed_speaker")
    ]
    backend_rejected_misses = sum(
        1
        for record in labelled
        if record.get("feedback") == "missed_speaker" and not record.get("accepted")
    )
    actionable_misses = sum(
        1
        for record in labelled
        if record.get("feedback") == "missed_speaker"
        and record.get("accepted")
        and (record.get("user_id") or record.get("candidate_user_id"))
        == record.get("actual_user_id")
    )

    current = simulate_threshold(labelled, current_threshold)
    result: dict[str, Any] = {
        "labelled_count": len(labelled),
        "minimum_labelled": MIN_LABELLED_DECISIONS,
        "ready": len(labelled) >= MIN_LABELLED_DECISIONS,
        "current_threshold": round(float(current_threshold), 2),
        "current_metrics": asdict(current),
        "backend_rejected_misses": backend_rejected_misses,
        "threshold_actionable_misses": actionable_misses,
        "false_accept_weight": FALSE_ACCEPT_WEIGHT,
        "missed_speaker_weight": MISSED_SPEAKER_WEIGHT,
        "recommended_threshold": None,
        "recommended_metrics": None,
    }
    if not result["ready"]:
        return result

    simulations = [
        simulate_threshold(labelled, step * _THRESHOLD_STEP)
        for step in range(0, 101)
    ]
    best_score = min(item.score for item in simulations)
    best_options = [item for item in simulations if item.score == best_score]
    # Avoid gratuitous changes: choose the best-performing threshold nearest the
    # current setting. If two are equally near, prefer the higher/safer value.
    best = min(
        best_options,
        key=lambda item: (abs(item.threshold - current_threshold), -item.threshold),
    )
    result["recommended_threshold"] = best.threshold
    result["recommended_metrics"] = asdict(best)
    return result
