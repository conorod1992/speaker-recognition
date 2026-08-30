"""Contract tests for opt-in production DSP and hidden neural capability."""

from pathlib import Path

ROOT = Path(__file__).parents[1]
COMPONENT = ROOT / "custom_components" / "speaker_recognition"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_stt_dsp_is_explicit_opt_in_and_defaults_off() -> None:
    const = _text(COMPONENT / "const.py")
    config_flow = _text(COMPONENT / "config_flow.py")

    assert 'CONF_USE_BASIC_DSP = "use_basic_dsp"' in const
    assert "DEFAULT_USE_BASIC_DSP = False" in const
    assert "Use basic DSP for speech-to-text" in _text(
        COMPONENT / "translations" / "en.json"
    )
    assert "selector.BooleanSelector()" in config_flow


def test_dsp_only_wraps_downstream_stt_not_speaker_or_whisper_analysis() -> None:
    stt = _text(COMPONENT / "stt.py")

    assert "stt_stream = async_enhance_stt_stream(" in stt
    assert "source_entity.async_process_audio_stream(\n                    metadata, stt_stream" in stt
    assert "prepare_live_pcm,\n                    audio_data" in stt
    assert "self.recognition.async_recognize(\n                        pcm_audio" in stt
    assert "detect_whisper,\n                            pcm_audio" in stt


def test_rnnoise_is_not_invoked_or_exposed_by_live_diagnostics() -> None:
    websocket = _text(COMPONENT / "enhancement_websocket.py")
    frontend = _text(
        COMPONENT / "www" / "speaker-recognition-enhancement-panel.js"
    )

    assert "async_denoise" not in websocket
    assert "RNNoise" not in frontend
    assert "rnnoise" not in frontend.lower()
    assert "Basic DSP" in frontend


def test_backend_neural_capability_is_retained_for_future_use() -> None:
    backend_api = _text(ROOT / "speaker_recognition" / "api.py")

    assert "/denoise" in backend_api
