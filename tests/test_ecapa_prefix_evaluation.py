"""Regression coverage for ECAPA early-audio duration experiments."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / "speaker_recognition"


def _analysis_module():
    path = INTEGRATION / "live_evaluation_analysis.py"
    spec = importlib.util.spec_from_file_location("prefix_evaluation_analysis_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _engine(score: float, *, backend: float = 0.2, effective: float = 0.0) -> dict:
    return {
        "engine_id": "ecapa_tdnn",
        "candidate_user_id": "alice",
        "similarity": score,
        "margin": None,
        "all_scores": {"alice": score},
        "backend_processing_seconds": backend,
        "call_seconds": backend + 0.02,
        "effective_added_latency_seconds": effective,
        "effective_added_latency_upper_bound": False,
    }


def _authoritative(score: float) -> dict:
    return {
        **_engine(score, backend=0.1),
        "engine_id": "resemblyzer",
    }


def test_prefix_analysis_uses_same_explicit_ground_truth() -> None:
    """A short ECAPA view can win or lose independently without defining truth."""
    module = _analysis_module()
    records = [
        {
            "actual_user_id": "alice",
            "authoritative": _authoritative(0.8),
            "shadow": _engine(0.8),
            "shadow_prefixes": {
                "1.0": _engine(0.15),
                "2.0": _engine(0.75),
                "2.5": _engine(0.8),
            },
        },
        {
            "actual_user_id": None,
            "authoritative": _authoritative(0.8),
            "shadow": _engine(0.2),
            "shadow_prefixes": {
                "1.0": _engine(0.7),
                "2.0": _engine(0.2),
                "2.5": _engine(0.2),
            },
        },
    ]

    result = module.analyze_live_evaluation(records)

    assert result["trial_count"] == 2
    assert result["shadow_prefixes"]["1.0"]["trials"] == 2
    assert result["shadow_prefixes"]["2.0"]["trials"] == 2
    assert result["shadow_prefixes"]["2.5"]["trials"] == 2
    assert result["shadow_prefixes"]["2.0"]["correct"] == 2
    assert result["shadow_prefixes"]["2.5"]["correct"] == 2


def test_short_utterances_are_excluded_only_from_unavailable_prefixes() -> None:
    """A 1-second-capable turn still contributes when 2/2.5 seconds are unavailable."""
    module = _analysis_module()
    record = {
        "actual_user_id": "alice",
        "authoritative": _authoritative(0.8),
        "shadow": _engine(0.8),
        "shadow_prefixes": {"1.0": _engine(0.7)},
    }

    result = module.analyze_live_evaluation([record])

    assert result["shadow_prefixes"]["1.0"]["trials"] == 1
    assert result["shadow_prefixes"]["2.0"] is None
    assert result["shadow_prefixes"]["2.5"] is None
    assert result["shadow"]["trials"] == 1


def test_live_evaluator_uses_fixed_prefixes_and_projects_early_start_latency() -> None:
    source = (INTEGRATION / "live_evaluation.py").read_text(encoding="utf-8")

    assert "PREFIX_DURATIONS_SECONDS = (1.0, 2.0, 2.5)" in source
    assert "def _prefix_pcm" in source
    assert 'current["shadow_prefixes"] = prefixes' in source
    assert "prefix_seconds + call_seconds - stt_seconds" in source
    assert 'engine["projected_early_start"] = True' in source
    assert "await _score(pcm_audio)" in source


def test_evaluation_ui_surfaces_duration_table_and_per_turn_prefix_diagnostics() -> None:
    frontend = (
        INTEGRATION / "www" / "speaker-recognition-evaluation-panel.js"
    ).read_text(encoding="utf-8")

    assert "ECAPA audio-length experiment" in frontend
    assert "Eligible labelled trials" in frontend
    assert "1.0 s" in frontend
    assert "2.0 s" in frontend
    assert "2.5 s" in frontend
    assert "Projected Assist latency" in frontend
    assert "only utterances long enough" in frontend
    assert "ECAPA early-audio diagnostics" in frontend


def test_updated_evaluation_modules_compile() -> None:
    for name in ("live_evaluation.py", "live_evaluation_analysis.py"):
        path = INTEGRATION / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
