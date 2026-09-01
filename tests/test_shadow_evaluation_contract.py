"""Contracts for paired authoritative/shadow evaluation in the HA integration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_calibration_module():
    path = Path("custom_components/speaker_recognition/calibration.py")
    spec = importlib.util.spec_from_file_location("speaker_recognition_calibration_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _paired_record(index: int) -> dict:
    return {
        "decision_id": str(index),
        "feedback": "correct",
        "actual_user_id": None,
        "identity_eligible": True,
        "user_id": "alice",
        "candidate_user_id": "alice",
        "engine_id": "resemblyzer",
        "all_scores": {"alice": 0.55, "bob": 0.50},
        "backend_processing_seconds": 0.08 + (index % 3) * 0.01,
        "shadow_engine_id": "ecapa_tdnn",
        "shadow_candidate_user_id": "alice",
        "shadow_similarity": 0.90,
        "shadow_margin": 0.80,
        "shadow_all_scores": {"alice": 0.90, "bob": 0.10},
        "shadow_processing_seconds": 0.20 + (index % 3) * 0.01,
    }


def test_shadow_comparison_uses_paired_turns_and_independent_operating_points() -> None:
    calibration = _load_calibration_module()
    result = calibration.analyze_engine_comparison(
        [_paired_record(index) for index in range(15)]
    )

    assert result["ready"] is True
    assert result["paired_count"] == 15
    assert result["coverage"] == 1.0
    assert result["authoritative"]["engine_id"] == "resemblyzer"
    assert result["shadow"]["engine_id"] == "ecapa_tdnn"
    assert result["authoritative"]["similarity_threshold"] != result["shadow"]["similarity_threshold"]
    assert result["authoritative"]["median_latency_seconds"] == 0.09
    assert result["shadow"]["median_latency_seconds"] == 0.21


def test_shadow_integration_stays_off_assist_critical_path() -> None:
    setup_source = Path("custom_components/speaker_recognition/__init__.py").read_text(
        encoding="utf-8"
    )
    evaluation_source = Path(
        "custom_components/speaker_recognition/shadow_evaluation.py"
    ).read_text(encoding="utf-8")
    recognition_source = Path(
        "custom_components/speaker_recognition/recognition.py"
    ).read_text(encoding="utf-8")

    assert "async_setup_shadow_evaluation(hass, history)" in setup_source
    assert "hass.async_create_task(" in evaluation_source
    assert "async_shadow_recognize" in evaluation_source
    assert '"/shadow/recognize"' in recognition_source
    assert "SHADOW_RECOGNITION_TIMEOUT_SECONDS" in recognition_source
