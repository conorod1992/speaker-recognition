"""Analyze labelled recognition history and recommend conservative settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable

MIN_LABELLED_DECISIONS = 15
FALSE_ACCEPT_WEIGHT = 5
MISSED_SPEAKER_WEIGHT = 1
_THRESHOLD_STEP = 0.01
_ENGINE_MARGIN_MAX = 0.50


@dataclass(frozen=True)
class ThresholdMetrics:
    """Simulated outcomes for one Home Assistant confidence threshold."""

    threshold: float
    correct_identity: int
    correct_rejection: int
    false_accepts: int
    missed_speakers: int
    score: int


@dataclass(frozen=True)
class _EngineTrial:
    """One paired labelled raw-score trial for engine comparison."""

    actual_user_id: str | None
    candidate_user_id: str
    similarity: float
    margin: float | None
    latency_seconds: float | None


@dataclass(frozen=True)
class EngineMetrics:
    """Open-set outcomes at one engine-specific operating point."""

    engine_id: str
    similarity_threshold: float
    margin_threshold: float
    trials: int
    correct_identity: int
    correct_rejection: int
    false_unknowns: int
    wrong_speaker: int
    false_accepts: int
    score: int
    median_latency_seconds: float | None

    @property
    def correct(self) -> int:
        return self.correct_identity + self.correct_rejection

    @property
    def false_identifications(self) -> int:
        return self.wrong_speaker + self.false_accepts


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
    """Return conservative HA-threshold guidance from explicitly labelled decisions."""
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
    best = min(
        best_options,
        key=lambda item: (abs(item.threshold - current_threshold), -item.threshold),
    )
    result["recommended_threshold"] = best.threshold
    result["recommended_metrics"] = asdict(best)
    return result


def _actual_user_for_engine_trial(record: dict[str, Any]) -> str | None:
    """Translate existing feedback semantics into open-set ground truth."""
    feedback = record.get("feedback")
    if feedback == "correct":
        if record.get("identity_eligible") and isinstance(record.get("user_id"), str):
            return str(record["user_id"])
        return None
    if feedback in ("wrong_speaker", "missed_speaker"):
        actual = record.get("actual_user_id")
        return actual if isinstance(actual, str) and actual else None
    raise ValueError("Decision is not labelled")


def _trial_from_scores(
    record: dict[str, Any],
    *,
    scores_key: str,
    latency_key: str,
) -> _EngineTrial | None:
    scores = record.get(scores_key)
    if not isinstance(scores, dict) or not scores:
        return None
    normalized = {
        str(user): float(score)
        for user, score in scores.items()
        if isinstance(user, str) and isinstance(score, (int, float))
    }
    if not normalized:
        return None
    ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    candidate, similarity = ranked[0]
    margin = similarity - ranked[1][1] if len(ranked) > 1 else None
    latency = record.get(latency_key)
    return _EngineTrial(
        actual_user_id=_actual_user_for_engine_trial(record),
        candidate_user_id=candidate,
        similarity=float(similarity),
        margin=float(margin) if margin is not None else None,
        latency_seconds=(float(latency) if isinstance(latency, (int, float)) else None),
    )


def _evaluate_engine(
    engine_id: str,
    trials: list[_EngineTrial],
    similarity_threshold: float,
    margin_threshold: float,
) -> EngineMetrics:
    correct_identity = 0
    correct_rejection = 0
    false_unknowns = 0
    wrong_speaker = 0
    false_accepts = 0
    latencies: list[float] = []

    for trial in trials:
        accepted = trial.similarity >= similarity_threshold and (
            trial.margin is None or trial.margin >= margin_threshold
        )
        predicted = trial.candidate_user_id if accepted else None
        if trial.latency_seconds is not None and trial.latency_seconds >= 0:
            latencies.append(trial.latency_seconds)

        if trial.actual_user_id is None:
            if predicted is None:
                correct_rejection += 1
            else:
                false_accepts += 1
        elif predicted == trial.actual_user_id:
            correct_identity += 1
        elif predicted is None:
            false_unknowns += 1
        else:
            wrong_speaker += 1

    score = FALSE_ACCEPT_WEIGHT * (wrong_speaker + false_accepts) + (
        MISSED_SPEAKER_WEIGHT * false_unknowns
    )
    return EngineMetrics(
        engine_id=engine_id,
        similarity_threshold=round(similarity_threshold, 2),
        margin_threshold=round(margin_threshold, 2),
        trials=len(trials),
        correct_identity=correct_identity,
        correct_rejection=correct_rejection,
        false_unknowns=false_unknowns,
        wrong_speaker=wrong_speaker,
        false_accepts=false_accepts,
        score=score,
        median_latency_seconds=median(latencies) if latencies else None,
    )


def _best_engine_operating_point(
    engine_id: str, trials: list[_EngineTrial]
) -> EngineMetrics:
    """Optimize similarity and margin independently for one engine's score scale."""
    best: EngineMetrics | None = None
    for similarity_step in range(0, 101):
        similarity_threshold = similarity_step * _THRESHOLD_STEP
        for margin_step in range(0, int(_ENGINE_MARGIN_MAX / _THRESHOLD_STEP) + 1):
            margin_threshold = margin_step * _THRESHOLD_STEP
            metrics = _evaluate_engine(
                engine_id,
                trials,
                similarity_threshold,
                margin_threshold,
            )
            if best is None or (
                metrics.score,
                -metrics.correct,
                -metrics.similarity_threshold,
                -metrics.margin_threshold,
            ) < (
                best.score,
                -best.correct,
                -best.similarity_threshold,
                -best.margin_threshold,
            ):
                best = metrics
    assert best is not None
    return best


def analyze_engine_comparison(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Compare authoritative and shadow engines on identical labelled turns."""
    labelled = [
        dict(record)
        for record in records
        if record.get("feedback") in ("correct", "wrong_speaker", "missed_speaker")
    ]
    paired: list[tuple[_EngineTrial, _EngineTrial, dict[str, Any]]] = []
    for record in labelled:
        authoritative = _trial_from_scores(
            record,
            scores_key="all_scores",
            latency_key="backend_processing_seconds",
        )
        shadow = _trial_from_scores(
            record,
            scores_key="shadow_all_scores",
            latency_key="shadow_processing_seconds",
        )
        if authoritative is not None and shadow is not None:
            paired.append((authoritative, shadow, record))

    shadow_ids = {
        str(record.get("shadow_engine_id"))
        for _, _, record in paired
        if isinstance(record.get("shadow_engine_id"), str)
    }
    authoritative_ids = {
        str(record.get("engine_id", "resemblyzer"))
        for _, _, record in paired
        if isinstance(record.get("engine_id", "resemblyzer"), str)
    }
    result: dict[str, Any] = {
        "labelled_count": len(labelled),
        "paired_count": len(paired),
        "minimum_labelled": MIN_LABELLED_DECISIONS,
        "ready": len(paired) >= MIN_LABELLED_DECISIONS,
        "coverage": (len(paired) / len(labelled) if labelled else 0.0),
        "false_identification_weight": FALSE_ACCEPT_WEIGHT,
        "false_unknown_weight": MISSED_SPEAKER_WEIGHT,
        "authoritative": None,
        "shadow": None,
    }
    if not paired or len(shadow_ids) != 1 or len(authoritative_ids) != 1:
        return result

    authoritative_id = next(iter(authoritative_ids))
    shadow_id = next(iter(shadow_ids))
    authoritative_trials = [item[0] for item in paired]
    shadow_trials = [item[1] for item in paired]

    # Operating points are intentionally optimized independently. A cosine score
    # from Resemblyzer is not numerically interchangeable with one from ECAPA.
    result["authoritative"] = asdict(
        _best_engine_operating_point(authoritative_id, authoritative_trials)
    )
    result["shadow"] = asdict(_best_engine_operating_point(shadow_id, shadow_trials))
    return result
