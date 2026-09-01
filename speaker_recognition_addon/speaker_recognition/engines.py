"""Speaker embedding engine abstractions."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
import resemblyzer  # type: ignore[import-untyped]

from speaker_recognition.const import DEFAULT_ENGINE_ID
from speaker_recognition.models import AudioInput


@dataclass(frozen=True)
class EngineInfo:
    """Stable metadata describing a speaker embedding engine."""

    engine_id: str
    display_name: str


class SpeakerEmbeddingEngine(Protocol):
    """Contract implemented by speaker embedding backends."""

    @property
    def info(self) -> EngineInfo:
        """Return stable engine metadata."""

    def prepare_audio(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Convert an API audio input into the engine's prepared waveform."""

    def embed_prepared(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Create one speaker embedding from an engine-prepared waveform."""


class ResemblyzerEngine:
    """Resemblyzer speaker embedding engine."""

    info = EngineInfo(engine_id=DEFAULT_ENGINE_ID, display_name="Resemblyzer")

    def __init__(self) -> None:
        """Initialize the pretrained Resemblyzer encoder."""
        self._encoder: Any = resemblyzer.VoiceEncoder()

    @property
    def encoder(self) -> Any:
        """Return the underlying encoder for transitional compatibility/tests."""
        return self._encoder

    @encoder.setter
    def encoder(self, value: Any) -> None:
        self._encoder = value

    def prepare_audio(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Decode PCM16 input and apply Resemblyzer preprocessing."""
        audio_bytes = base64.b64decode(audio_input.audio_data)
        audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16).copy()
        if audio_array_int16.size == 0:
            raise ValueError("Empty audio data")
        audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
        if float(np.max(np.abs(audio_array_float32))) < 1e-5:
            raise ValueError("Audio data contains no usable speech signal")
        result: NDArray[np.float32] = resemblyzer.preprocess_wav(
            audio_array_float32, source_sr=audio_input.sample_rate
        )
        return result

    def embed_prepared(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return a Resemblyzer embedding for prepared audio."""
        return np.asarray(self._encoder.embed_utterance(waveform), dtype=np.float32)


_ENGINE_FACTORIES = {
    DEFAULT_ENGINE_ID: ResemblyzerEngine,
}


def available_engines() -> tuple[EngineInfo, ...]:
    """Return metadata for engines available in this service build."""
    return tuple(factory.info for factory in _ENGINE_FACTORIES.values())


def create_engine(engine_id: str = DEFAULT_ENGINE_ID) -> SpeakerEmbeddingEngine:
    """Create a configured speaker embedding engine by stable ID."""
    factory = _ENGINE_FACTORIES.get(engine_id)
    if factory is None:
        raise ValueError(f"Unknown speaker embedding engine: {engine_id}")
    return factory()
