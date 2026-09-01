"""Source-level contracts for ECAPA shadow API isolation."""

from pathlib import Path


def test_backend_exposes_separate_shadow_endpoints_and_lock() -> None:
    source = Path("speaker_recognition/api.py").read_text(encoding="utf-8")
    assert "_SHADOW_LOCK = Lock()" in source
    assert '"/shadow/train"' in source
    assert '"/shadow/profiles/sync"' in source
    assert '"/shadow/recognize"' in source
    assert "with _RECOGNIZER_LOCK:" in source
    assert "with _SHADOW_LOCK:" in source


def test_addon_defaults_shadow_engine_off() -> None:
    config = Path("speaker_recognition_addon/config.yaml").read_text(encoding="utf-8")
    run = Path(
        "speaker_recognition_addon/rootfs/etc/s6-overlay/s6-rc.d/speaker-recognition/run"
    ).read_text(encoding="utf-8")
    assert 'shadow_engine: "none"' in config
    assert "list(none|ecapa_tdnn)" in config
    assert 'export SHADOW_ENGINE="none"' in run
