"""Regression coverage for the opt-in ECAPA shadow engine."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from speaker_recognition.const import DEFAULT_SHADOW_ENGINE, ECAPA_ENGINE_ID
from speaker_recognition.engines import EcapaTdnnEngine, available_engines
from speaker_recognition.models import AudioInput, Config


def _pcm_sine(sample_rate: int, seconds: float = 0.2) -> bytes:
    samples = np.arange(int(sample_rate * seconds), dtype=np.float32)
    waveform = 0.1 * np.sin(2 * np.pi * 220.0 * samples / sample_rate)
    return (waveform * 32767.0).astype(np.int16).tobytes()


def test_ecapa_engine_is_known_but_not_default() -> None:
    """ECAPA is discoverable while normal installations remain lightweight."""
    config = Config()
    assert config.shadow_engine == DEFAULT_SHADOW_ENGINE == "none"
    engines = {item.engine_id: item for item in available_engines()}
    assert ECAPA_ENGINE_ID in engines
    assert engines[ECAPA_ENGINE_ID].experimental is True


def test_ecapa_preparation_does_not_load_or_download_model(tmp_path: Path) -> None:
    """Audio preparation must remain local until an actual shadow embedding is needed."""
    engine = EcapaTdnnEngine(str(tmp_path))
    audio = AudioInput(
        audio_data=base64.b64encode(_pcm_sine(8000)).decode("ascii"),
        sample_rate=8000,
    )

    prepared = engine.prepare_audio(audio)

    assert engine._classifier is None
    assert prepared.dtype == np.float32
    assert 3000 <= prepared.size <= 3400
    assert not (tmp_path / ECAPA_ENGINE_ID).exists()


def test_ecapa_runtime_import_is_lazy() -> None:
    """Importing the service/test suite must not require SpeechBrain to be installed."""
    source = Path("speaker_recognition/engines.py").read_text(encoding="utf-8")
    class_start = source.index("class EcapaTdnnEngine")
    load_start = source.index("    def _load_classifier", class_start)
    speechbrain_import = source.index("from speechbrain.inference.classifiers", load_start)
    assert speechbrain_import > load_start
