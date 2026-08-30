"""Contract coverage for the dormant optional RNNoise capability."""

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
BACKEND = ROOT / "speaker_recognition"
ADDON_BACKEND = ROOT / "speaker_recognition_addon" / "speaker_recognition"
HA = ROOT / "custom_components" / "speaker_recognition"


def _load_neural_module():
    module_path = BACKEND / "neural_denoise.py"
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_neural_denoise_test", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_backend_source_copy_and_addon_include_rnnoise() -> None:
    """The dormant RNNoise implementation and native runtime remain available."""
    assert (BACKEND / "neural_denoise.py").read_text(encoding="utf-8") == (
        ADDON_BACKEND / "neural_denoise.py"
    ).read_text(encoding="utf-8")
    assert (BACKEND / "api.py").read_text(encoding="utf-8") == (
        ADDON_BACKEND / "api.py"
    ).read_text(encoding="utf-8")
    assert (BACKEND / "models.py").read_text(encoding="utf-8") == (
        ADDON_BACKEND / "models.py"
    ).read_text(encoding="utf-8")

    addon_dockerfile = (ROOT / "speaker_recognition_addon" / "Dockerfile").read_text(
        encoding="utf-8"
    )
    assert "pyrnnoise==0.4.3" in addon_dockerfile
    assert "--no-deps" in addon_dockerfile

    neural_source = (BACKEND / "neural_denoise.py").read_text(encoding="utf-8")
    assert 'metadata.distribution("pyrnnoise")' in neural_source
    assert "ctypes.CDLL" in neural_source
    assert "from pyrnnoise" not in neural_source

    root_dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "pyrnnoise" not in root_dockerfile


def test_rnnoise_is_dormant_and_not_user_facing() -> None:
    """RNNoise stays available in source without a runtime/API/frontend path."""
    api = (BACKEND / "api.py").read_text(encoding="utf-8")
    recognition = (HA / "recognition.py").read_text(encoding="utf-8")
    websocket = (HA / "enhancement_websocket.py").read_text(encoding="utf-8")
    panel = (HA / "www" / "speaker-recognition-enhancement-panel.js").read_text(
        encoding="utf-8"
    )

    assert '"/denoise"' not in api
    assert "denoise_pcm_rnnoise" not in api
    assert '"/denoise"' in recognition  # Capability retained in the HA client only.
    assert "async_denoise" not in websocket
    assert "RNNoise" not in panel
    assert "rnnoise" not in panel.lower()
    assert "Original from Home Assistant" in panel
    assert "Basic DSP" in panel


def test_resampling_round_trip_keeps_expected_shape() -> None:
    """The retained RNNoise 48 kHz bridge preserves practical utterance length."""
    module = _load_neural_module()
    original = np.arange(16000, dtype=np.int16)
    at_48k = module._resample_int16(original, 16000, 48000)
    restored = module._resample_int16(at_48k, 48000, 16000)

    assert abs(at_48k.size - 48000) <= 1
    assert abs(restored.size - original.size) <= 1
    assert restored.dtype == np.int16
