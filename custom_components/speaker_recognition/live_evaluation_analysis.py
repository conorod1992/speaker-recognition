"""Analysis helpers for explicitly labelled live engine A/B trials."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

FALSE_IDENTIFICATION_WEIGHT = 5
FALSE_UNKNOWN_WEIGHT = 1
_SIMILARITY_STEP = 0.01
_MARGIN_MAX = 0.50
_PREFIX_KEYS = ("1.0", "2.0", "2.5")


@dataclass(frozen=True)
class _Trial:
    actual_user_id: str | None
    scores: dict[str, float]
    backend_seconds: float | None
    call_seconds: float | None
    effective_added_seconds: float | None
    effective_is_upper_bound: bool


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _trial_from_engine(
    record: dict[str, Any], engine: Any
) -> tuple[str, _Trial] | None:
    if not isinstance(engine, dict):
        return None
    engine_id = engine.get("engine_id")
    scores = engine.get("all_scores")
    if not isinstance(engine_id, str) or not engine_id or not isinstance(scores, dict):
        return None
    normalized = {
        str(user): float(score)
        for user, score in scores.items()
        if isinstance(user, str) and isinstance(score, (int, float))
    }
    if not normalized:
        return None
    actual = record.get("actual_user_id")
    actual_user_id = actual if isinstance(actual, str) and actual else None
    return engine_id, _Trial(
        actual_user_id=actual_user_id,
        scores=normalized,
        backend_seconds=_number(engine.get("backend_processing_seconds")),
        call_seconds=_number(engine.get("call_seconds")),
        effective_added_seconds=_number(engine.get("effective_added_latency_seconds")),
        effective_is_upper_bound=bool(engine.get("effective_added_latency_upper_bound")),
    )


def _trial(record: dict[str, Any], side: str) -> tuple[str, _Trial] | None:
    return _trial_from_engine(record, record.get(side))


def _decision(
    scores: dict[str, float], similarity_threshold: float, margin_threshold: float
) -> str | None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return None
    candidate, similarity = ranked[0]
    margin = similarity - ranked[1][1] if len(ranked) > 1 else None
    if similarity < similarity_threshold:
        return None
    if margin is not None and margin < margin_threshold:
        return None
    return candidate


def _evaluate(
    engine_id: str,
    trials: list[_Trial],
    similarity_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    correct_identity = 0
    correct_rejection = 0
    false_unknowns = 0
    wrong_speaker = 0
    false_accepts = 0

    for trial in trials:
        predicted = _decision(trial.scores, similarity_threshold, margin_threshold)
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

    false_identifications = wrong_speaker + false_accepts
    score = (
        FALSE_IDENTIFICATION_WEIGHT * false_identifications
        + FALSE_UNKNOWN_WEIGHT * false_unknowns
    )
    backend = [x.backend_seconds for x in trials if x.backend_seconds is not None]
    calls = [x.call_seconds for x in trials if x.call_seconds is not None]
    effective = [
        x.effective_added_seconds
        for x in trials
        if x.effective_added_seconds is not None
    ]
    known_trials = sum(x.actual_user_id is not None for x in trials)
    unknown_trials = len(trials) - known_trials
    return {
        "engine_id": engine_id,
        "trials": len(trials),
        "known_trials": known_trials,
        "unknown_trials": unknown_trials,
        "correct_identity": correct_identity,
        "correct_rejection": correct_rejection,
        "correct": correct_identity + correct_rejection,
        "false_unknowns": false_unknowns,
        "wrong_speaker": wrong_speaker,
        "false_accepts": false_accepts,
        "false_identifications": false_identifications,
        "score": score,
        "similarity_threshold": round(similarity_threshold, 2),
        "margin_threshold": round(margin_threshold, 2),
        "median_backend_seconds": median(backend) if backend else None,
        "median_call_seconds": median(calls) if calls else None,
        "median_effective_added_latency_seconds": median(effective) if effective else None,
        "effective_latency_contains_upper_bounds": any(
            x.effective_is_upper_bound for x in trials
        ),
    }


def _best(engine_id: str, trials: list[_Trial]) -> dict[str, Any]:
    margin_relevant = any(len(trial.scores) > 1 for trial in trials)
    margin_steps = (
        range(0, int(_MARGIN_MAX / _SIMILARITY_STEP) + 1)
        if margin_relevant
        else (0,)
    )
    best: dict[str, Any] | None = None
    for similarity_step in range(0, 101):
        similarity = similarity_step * _SIMILARITY_STEP
        for margin_step in margin_steps:
            margin = margin_step * _SIMILARITY_STEP
            metrics = _evaluate(engine_id, trials, similarity, margin)
            if best is None or (
                metrics["score"],
                -metrics["correct"],
                -metrics["similarity_threshold"],
                -metrics["margin_threshold"],
            ) < (
                best["score"],
                -best["correct"],
                -best["similarity_threshold"],
                -best["margin_threshold"],
            ):
                best = metrics
    assert best is not None
    best["margin_relevant"] = margin_relevant
    return best


def _prefix_analysis(
    records: list[dict[str, Any]], prefix_key: str
) -> dict[str, Any] | None:
    trials: list[tuple[str, _Trial]] = []
    for record in records:
        prefixes = record.get("shadow_prefixes")
        if not isinstance(prefixes, dict):
            continue
        trial = _trial_from_engine(record, prefixes.get(prefix_key))
        if trial is not None:
            trials.append(trial)
    if not trials:
        return None
    engine_ids = {item[0] for item in trials}
    if len(engine_ids) != 1:
        return None
    result = _best(next(iter(engine_ids)), [item[1] for item in trials])
    result["prefix_seconds"] = float(prefix_key)
    result["projected_early_start"] = True
    return result


def analyze_live_evaluation(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare both engines using only explicit live-test ground truth."""
    authoritative: list[tuple[str, _Trial]] = []
    shadow: list[tuple[str, _Trial]] = []
    for record in records:
        first = _trial(record, "authoritative")
        second = _trial(record, "shadow")
        if first is not None and second is not None:
            authoritative.append(first)
            shadow.append(second)

    result: dict[str, Any] = {
        "trial_count": len(authoritative),
        "false_identification_weight": FALSE_IDENTIFICATION_WEIGHT,
        "false_unknown_weight": FALSE_UNKNOWN_WEIGHT,
        "authoritative": None,
        "shadow": None,
        "shadow_prefixes": {
            prefix_key: _prefix_analysis(records, prefix_key)
            for prefix_key in _PREFIX_KEYS
        },
    }
    if not authoritative or len(authoritative) != len(shadow):
        return result

    authoritative_ids = {item[0] for item in authoritative}
    shadow_ids = {item[0] for item in shadow}
    if len(authoritative_ids) != 1 or len(shadow_ids) != 1:
        result["mixed_engines"] = True
        return result

    auth_trials = [item[1] for item in authoritative]
    shadow_trials = [item[1] for item in shadow]
    result["authoritative"] = _best(next(iter(authoritative_ids)), auth_trials)
    result["shadow"] = _best(next(iter(shadow_ids)), shadow_trials)
    result["known_trials"] = sum(x.actual_user_id is not None for x in auth_trials)
    result["unknown_trials"] = len(auth_trials) - result["known_trials"]
    return result
