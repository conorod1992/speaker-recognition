"""Offline/shadow evaluation helpers for speaker recognition engines."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Optional


@dataclass(frozen=True)
class EvaluationRecord:
    """One labelled recognition trial from one engine."""

    engine_id: str
    all_scores: Mapping[str, float]
    actual_user_id: Optional[str]
    trial_id: Optional[str] = None
    latency_seconds: Optional[float] = None


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate outcomes for one engine and decision policy."""

    engine_id: str
    trials: int
    known_trials: int
    unknown_trials: int
    correct_known: int
    correct_unknown: int
    false_unknowns: int
    wrong_speaker: int
    false_accepts: int
    median_latency_seconds: Optional[float]

    @property
    def correct(self) -> int:
        """Return the number of correct known and unknown decisions."""
        return self.correct_known + self.correct_unknown

    @property
    def false_identifications(self) -> int:
        """Return wrong-known-speaker plus unknown-speaker false accepts."""
        return self.wrong_speaker + self.false_accepts

    @property
    def accuracy(self) -> float:
        """Return total decision accuracy."""
        return self.correct / self.trials if self.trials else 0.0

    @property
    def false_identification_rate(self) -> float:
        """Return dangerous false identifications as a share of all trials."""
        return self.false_identifications / self.trials if self.trials else 0.0

    @property
    def false_unknown_rate(self) -> float:
        """Return missed known speakers as a share of known-speaker trials."""
        return self.false_unknowns / self.known_trials if self.known_trials else 0.0


@dataclass(frozen=True)
class EvaluationOperatingPoint:
    """One threshold/margin policy and its measured cost."""

    similarity_threshold: float
    margin_threshold: float
    cost: float
    metrics: EvaluationMetrics


def decision_from_scores(
    all_scores: Mapping[str, float],
    similarity_threshold: float,
    margin_threshold: float,
) -> Optional[str]:
    """Apply an open-set decision policy to raw per-user similarity scores."""
    if not all_scores:
        return None
    ranked = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
    candidate, best_score = ranked[0]
    margin = best_score - ranked[1][1] if len(ranked) > 1 else None
    if best_score < similarity_threshold:
        return None
    if margin is not None and margin < margin_threshold:
        return None
    return candidate


def evaluate_records(
    records: Iterable[EvaluationRecord],
    similarity_threshold: float,
    margin_threshold: float,
) -> EvaluationMetrics:
    """Evaluate labelled records for exactly one engine."""
    materialized = list(records)
    engine_ids = {record.engine_id for record in materialized}
    if len(engine_ids) > 1:
        raise ValueError("evaluate_records requires records from exactly one engine")
    engine_id = next(iter(engine_ids), "")

    known_trials = 0
    unknown_trials = 0
    correct_known = 0
    correct_unknown = 0
    false_unknowns = 0
    wrong_speaker = 0
    false_accepts = 0
    latencies: list[float] = []

    for record in materialized:
        predicted = decision_from_scores(
            record.all_scores, similarity_threshold, margin_threshold
        )
        if record.latency_seconds is not None and record.latency_seconds >= 0:
            latencies.append(record.latency_seconds)

        if record.actual_user_id is None:
            unknown_trials += 1
            if predicted is None:
                correct_unknown += 1
            else:
                false_accepts += 1
            continue

        known_trials += 1
        if predicted == record.actual_user_id:
            correct_known += 1
        elif predicted is None:
            false_unknowns += 1
        else:
            wrong_speaker += 1

    return EvaluationMetrics(
        engine_id=engine_id,
        trials=len(materialized),
        known_trials=known_trials,
        unknown_trials=unknown_trials,
        correct_known=correct_known,
        correct_unknown=correct_unknown,
        false_unknowns=false_unknowns,
        wrong_speaker=wrong_speaker,
        false_accepts=false_accepts,
        median_latency_seconds=median(latencies) if latencies else None,
    )


def compare_engines(
    records: Iterable[EvaluationRecord],
    similarity_threshold: float,
    margin_threshold: float,
) -> dict[str, EvaluationMetrics]:
    """Evaluate each engine independently under the same decision policy."""
    grouped: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        grouped.setdefault(record.engine_id, []).append(record)
    return {
        engine_id: evaluate_records(
            engine_records, similarity_threshold, margin_threshold
        )
        for engine_id, engine_records in sorted(grouped.items())
    }


def find_best_operating_point(
    records: Iterable[EvaluationRecord],
    similarity_thresholds: Iterable[float],
    margin_thresholds: Iterable[float],
    false_identification_weight: float = 5.0,
    false_unknown_weight: float = 1.0,
) -> EvaluationOperatingPoint:
    """Find the lowest-cost threshold pair for labelled records from one engine."""
    materialized = list(records)
    if not materialized:
        raise ValueError("At least one labelled evaluation record is required")
    candidates: list[EvaluationOperatingPoint] = []
    for similarity_threshold in similarity_thresholds:
        for margin_threshold in margin_thresholds:
            metrics = evaluate_records(
                materialized, similarity_threshold, margin_threshold
            )
            cost = (
                false_identification_weight * metrics.false_identifications
                + false_unknown_weight * metrics.false_unknowns
            )
            candidates.append(
                EvaluationOperatingPoint(
                    similarity_threshold=float(similarity_threshold),
                    margin_threshold=float(margin_threshold),
                    cost=float(cost),
                    metrics=metrics,
                )
            )
    if not candidates:
        raise ValueError("Threshold grids must not be empty")
    return min(
        candidates,
        key=lambda item: (
            item.cost,
            -item.metrics.correct,
            -item.similarity_threshold,
            -item.margin_threshold,
        ),
    )
