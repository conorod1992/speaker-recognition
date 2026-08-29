"""Tests for one-time speaker encoder startup warm-up."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np


def _load_warmup_module():
    path = Path(__file__).parents[1] / "speaker_recognition" / "warmup.py"
    spec = importlib.util.spec_from_file_location("speaker_recognition_warmup_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Encoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def embed_utterance(self, waveform: np.ndarray) -> np.ndarray:
        self.calls += 1
        assert waveform.dtype == np.float32
        assert waveform.size > 0
        if self.fail:
            raise RuntimeError("cold-start failure")
        return np.array([1.0, 0.5], dtype=np.float32)


class _Recognizer:
    def __init__(self, encoder: _Encoder) -> None:
        self._encoder = encoder


def test_warmup_runs_real_embedding(monkeypatch) -> None:
    warmup = _load_warmup_module()
    monkeypatch.delenv("SPEAKER_RECOGNITION_SKIP_WARMUP", raising=False)
    encoder = _Encoder()

    status = warmup.warm_encoder(_Recognizer(encoder))

    assert encoder.calls == 1
    assert status.ready is True
    assert status.error is None
    assert status.seconds >= 0.0


def test_warmup_failure_is_reported_without_crashing(monkeypatch) -> None:
    warmup = _load_warmup_module()
    monkeypatch.delenv("SPEAKER_RECOGNITION_SKIP_WARMUP", raising=False)
    encoder = _Encoder(fail=True)

    status = warmup.warm_encoder(_Recognizer(encoder))

    assert encoder.calls == 1
    assert status.ready is False
    assert "cold-start failure" in (status.error or "")


def test_image_build_can_skip_warmup(monkeypatch) -> None:
    warmup = _load_warmup_module()
    monkeypatch.setenv("SPEAKER_RECOGNITION_SKIP_WARMUP", "1")
    encoder = _Encoder()

    status = warmup.warm_encoder(_Recognizer(encoder))

    assert encoder.calls == 0
    assert status.ready is True
    assert status.seconds == 0.0
