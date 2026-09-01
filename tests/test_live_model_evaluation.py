"""Regression coverage for the dedicated live Resemblyzer/ECAPA evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
INTEGRATION = ROOT / "custom_components" / "speaker_recognition"


def _analysis_module():
    path = INTEGRATION / "live_evaluation_analysis.py"
    spec = importlib.util.spec_from_file_location("live_evaluation_analysis_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _engine(engine_id: str, score: float, *, backend: float = 0.1) -> dict:
    return {
        "engine_id": engine_id,
        "candidate_user_id": "alice",
        "similarity": score,
        "margin": None,
        "all_scores": {"alice": score},
        "backend_processing_seconds": backend,
        "call_seconds": backend + 0.02,
        "effective_added_latency_seconds": 0.0,
        "effective_added_latency_upper_bound": False,
    }


def test_live_analysis_uses_explicit_unknown_ground_truth() -> None:
    """The active model can lose a trial; its own answer never defines truth."""
    module = _analysis_module()
    records = [
        {
            "actual_user_id": "alice",
            "authoritative": _engine("resemblyzer", 0.80),
            "shadow": _engine("ecapa_tdnn", 0.80),
        },
        {
            "actual_user_id": None,
            "authoritative": _engine("resemblyzer", 0.80),
            "shadow": _engine("ecapa_tdnn", 0.20),
        },
    ]

    result = module.analyze_live_evaluation(records)

    assert result["trial_count"] == 2
    assert result["known_trials"] == 1
    assert result["unknown_trials"] == 1
    assert result["authoritative"]["correct"] == 1
    assert result["authoritative"]["score"] == 1
    assert result["shadow"]["correct"] == 2
    assert result["shadow"]["score"] == 0


def test_live_analysis_has_no_minimum_trial_gate() -> None:
    """Metrics update from the first explicit trial and keep accumulating."""
    module = _analysis_module()
    result = module.analyze_live_evaluation(
        [
            {
                "actual_user_id": "alice",
                "authoritative": _engine("resemblyzer", 0.70),
                "shadow": _engine("ecapa_tdnn", 0.60),
            }
        ]
    )

    assert result["trial_count"] == 1
    assert result["authoritative"] is not None
    assert result["shadow"] is not None


def test_live_evaluation_pairs_models_before_normal_assist_returns() -> None:
    """Shadow scoring starts beside the active call and is later bound to STT timing."""
    source = (INTEGRATION / "live_evaluation.py").read_text(encoding="utf-8")
    shadow = (INTEGRATION / "shadow_evaluation.py").read_text(encoding="utf-8")

    subclass = source.split("class LiveEvaluationSpeakerRecognition", 1)[1]
    assert subclass.index("evaluation.begin_pair") < subclass.index("super().async_recognize")
    assert "self.hass.async_create_task" in source
    assert "attach_assist_timing" in shadow
    assert "do not launch a duplicate" in shadow


def test_live_evaluation_is_persistent_until_explicit_clear() -> None:
    """There is no fifteen-trial cutoff or rolling history cap."""
    source = (INTEGRATION / "live_evaluation.py").read_text(encoding="utf-8")
    analysis = (INTEGRATION / "live_evaluation_analysis.py").read_text(encoding="utf-8")
    frontend = (
        INTEGRATION / "www" / "speaker-recognition-evaluation-panel.js"
    ).read_text(encoding="utf-8")

    assert 'f"{DOMAIN}.live_model_evaluation"' in source
    assert "self._records.append(record)" in source
    assert "self._records.clear()" in source
    assert "MAX_DECISIONS" not in source
    assert "MIN_LABELLED" not in analysis
    assert "Clear results" in frontend
    assert "no trial limit" in frontend.lower()
    assert "/ 15" not in frontend


def test_frontend_requests_independent_ground_truth() -> None:
    """The A/B UI asks who spoke instead of reusing Recognition History feedback."""
    frontend = (
        INTEGRATION / "www" / "speaker-recognition-evaluation-panel.js"
    ).read_text(encoding="utf-8")
    websocket = (INTEGRATION / "shadow_websocket.py").read_text(encoding="utf-8")

    assert "Was that" in frontend
    assert "someone not enrolled" in frontend
    assert "Discard trial" in frontend
    assert "speaker_recognition/evaluation_status" in frontend
    assert 'f"{DOMAIN}/evaluation_label"' in websocket
    assert 'vol.Required("actual_user_id"): vol.Any(str, None)' in websocket


def test_live_evaluation_reports_parallel_latency_not_just_backend_time() -> None:
    source = (INTEGRATION / "live_evaluation.py").read_text(encoding="utf-8")
    frontend = (
        INTEGRATION / "www" / "speaker-recognition-evaluation-panel.js"
    ).read_text(encoding="utf-8")

    assert "effective_added_latency_seconds" in source
    assert "recognition_seconds - preparation_seconds" in source
    assert "effective_added_latency_upper_bound" in source
    assert "Effective Assist latency" in frontend
    assert "STT and recognition running in parallel" in frontend


def test_new_evaluation_modules_compile() -> None:
    for name in (
        "live_evaluation.py",
        "live_evaluation_analysis.py",
        "shadow_evaluation.py",
        "shadow_websocket.py",
    ):
        path = INTEGRATION / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
