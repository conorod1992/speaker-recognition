"""Regression contracts for backend security and Assist fail-open behavior."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from speaker_recognition.const import MAX_TRAINING_SAMPLES
from speaker_recognition.models import AudioInput, TrainingRequest

ROOT = Path(__file__).parents[1]
BACKEND = ROOT / "speaker_recognition"
ADDON_BACKEND = ROOT / "speaker_recognition_addon" / "speaker_recognition"
HA = ROOT / "custom_components" / "speaker_recognition"


def test_security_sensitive_backend_sources_stay_synced() -> None:
    """The add-on must ship exactly the hardened canonical backend code."""
    for filename in ("__main__.py", "api.py", "const.py", "models.py"):
        canonical = (BACKEND / filename).read_text(encoding="utf-8")
        addon = (ADDON_BACKEND / filename).read_text(encoding="utf-8")
        assert canonical == addon


def test_remote_api_requires_authentication_by_default() -> None:
    """LAN callers cannot reach identity endpoints merely because port 8099 is open."""
    api = (BACKEND / "api.py").read_text(encoding="utf-8")
    addon_config = (ROOT / "speaker_recognition_addon" / "config.yaml").read_text(
        encoding="utf-8"
    )

    assert "dependencies=[Depends(require_api_access)]" in api
    assert "secrets.compare_digest" in api
    assert 'authorization.startswith("Bearer ")' in api
    assert 'authorization.startswith("Basic ")' in api
    assert 'os.environ.get("TRUSTED_LOCAL_HOSTS", "172.30.32.1")' in api
    assert "str(address) in _TRUSTED_LOCAL_HOSTS" in api
    assert "allow_insecure_remote: false" in addon_config
    assert "api_token: password" in addon_config


def test_addon_discovers_same_host_addresses_without_trusting_the_whole_lan() -> None:
    """HAOS published-port calls trust only addresses Supervisor says belong to HA."""
    addon = ROOT / "speaker_recognition_addon"
    run = (
        addon
        / "rootfs"
        / "etc"
        / "s6-overlay"
        / "s6-rc.d"
        / "speaker-recognition"
        / "run"
    ).read_text(encoding="utf-8")
    config = (addon / "config.yaml").read_text(encoding="utf-8")

    assert "hassio_api: true" in config
    assert "hassio_role:" not in config
    assert "http://supervisor/network/info" in run
    assert "SUPERVISOR_TOKEN" in run
    assert 'hosts = {"172.30.32.1"}' in run
    assert "ipaddress.ip_interface(value).ip" in run
    assert "export TRUSTED_LOCAL_HOSTS" in run
    assert "Trusted local hosts:" in run
    assert "192.168.0.0/16" not in run
    assert "is_private" not in run


def test_authentication_logging_matches_dynamic_local_trust() -> None:
    """Startup logs must not claim an add-on with trusted HA hosts is loopback-only."""
    main = (BACKEND / "__main__.py").read_text(encoding="utf-8")
    run = (
        ROOT
        / "speaker_recognition_addon"
        / "rootfs"
        / "etc"
        / "s6-overlay"
        / "s6-rc.d"
        / "speaker-recognition"
        / "run"
    ).read_text(encoding="utf-8")

    assert "trusted local hosts only" in main
    assert "loopback only" not in main
    assert "trusted local callers only" in run


def test_addon_smoke_exercises_authenticated_health() -> None:
    """CI verifies the secured add-on through the same published-port boundary."""
    quality = (ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )

    assert '"api_token":"smoke-api-token"' in quality
    assert "--user 'smoke-api-token:'" in quality


def test_recognition_has_short_fail_open_deadline() -> None:
    """A stalled recognition backend cannot hold a completed Assist turn for minutes."""
    recognition = (HA / "recognition.py").read_text(encoding="utf-8")

    assert "RECOGNITION_TIMEOUT_SECONDS = 4.0" in recognition
    assert "response = await asyncio.wait_for(" in recognition
    assert "timeout=RECOGNITION_TIMEOUT_SECONDS" in recognition
    assert "except asyncio.TimeoutError:" in recognition
    assert "continuing Assist without identity" in recognition


def test_health_retries_failed_warmup_and_container_checks_status() -> None:
    """Health can recover transient warm-up failures and degraded is not Docker-healthy."""
    api = (BACKEND / "api.py").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "if not _WARMUP_STATUS.ready:" in api
    assert "_WARMUP_STATUS = warm_encoder(recognizer)" in api
    assert "data.get('status') == 'healthy'" in dockerfile


def test_audio_sample_rate_is_bounded() -> None:
    """Implausible sample rates are rejected before audio processing."""
    with pytest.raises(ValidationError):
        AudioInput(audio_data="AAAA", sample_rate=1)
    with pytest.raises(ValidationError):
        AudioInput(audio_data="AAAA", sample_rate=192000)


def test_training_request_count_is_bounded() -> None:
    """One request cannot enqueue an unbounded number of training samples."""
    sample = {
        "user": "alice",
        "audio": {"audio_data": "AAAA", "sample_rate": 16000},
    }
    with pytest.raises(ValidationError):
        TrainingRequest(voice_samples=[sample] * (MAX_TRAINING_SAMPLES + 1))
