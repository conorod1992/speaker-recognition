"""Contracts for secure remote backends and memory budgets."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from speaker_recognition.const import MAX_AUDIO_BASE64_CHARS
from speaker_recognition.models import AudioInput, TrainingRequest, VoiceSample

ROOT = Path(__file__).parents[1]
HA = ROOT / "custom_components" / "speaker_recognition"


def test_training_request_has_aggregate_audio_budget() -> None:
    # Each sample remains within the existing 12 MiB decoded per-sample budget,
    # while six maximum-sized samples exceed the 64 MiB aggregate budget.
    encoded = "A" * MAX_AUDIO_BASE64_CHARS
    samples = [
        VoiceSample(
            user=f"user-{index}",
            audio=AudioInput(audio_data=encoded, sample_rate=16000),
        )
        for index in range(6)
    ]
    with pytest.raises(ValidationError, match="Combined training audio"):
        TrainingRequest(voice_samples=samples)


def test_remote_backend_token_is_first_class_ha_configuration() -> None:
    const = (HA / "const.py").read_text(encoding="utf-8")
    recognition = (HA / "recognition.py").read_text(encoding="utf-8")
    config_flow = (HA / "config_flow.py").read_text(encoding="utf-8")
    assert 'CONF_BACKEND_TOKEN = "backend_token"' in const
    assert '"Authorization": f"Bearer {self._api_token}"' in recognition
    assert "CONF_BACKEND_TOKEN" in config_flow


def test_local_media_reads_are_stat_first_and_bounded() -> None:
    audio = (HA / "audio.py").read_text(encoding="utf-8")
    recognition = (HA / "recognition.py").read_text(encoding="utf-8")
    config_flow = (HA / "config_flow.py").read_text(encoding="utf-8")
    assert "path.stat().st_size > MAX_LOCAL_WAV_BYTES" in audio
    assert "stream.read(MAX_LOCAL_WAV_BYTES + 1)" in audio
    assert "read_bounded_wav" in recognition
    assert "read_bounded_wav" in config_flow


def test_backend_has_transport_body_limit() -> None:
    api = (ROOT / "speaker_recognition" / "api.py").read_text(encoding="utf-8")
    assert "RequestBodyLimitMiddleware" in api
    assert "MAX_REQUEST_BODY_BYTES" in api
