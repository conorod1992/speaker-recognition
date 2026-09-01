"""Speaker embedding engine abstractions."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
import resemblyzer  # type: ignore[import-untyped]
from scipy.signal import resample_poly

from speaker_recognition.const import (
    DEFAULT_ENGINE_ID,
    DEFAULT_MODEL_CACHE_DIR,
    ECAPA_ENGINE_ID,
)
from speaker_recognition.models import AudioInput

ECAPA_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
# Pin the public model repository so a future upstream change cannot silently
# alter comparison results for the same application release.
ECAPA_MODEL_REVISION = "0f99f2d"
ECAPA_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class EngineInfo:
    """Stable metadata describing a speaker embedding engine."""

    engine_id: str
    display_name: str
    experimental: bool = False


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


class EcapaTdnnEngine:
    """Lazy SpeechBrain ECAPA-TDNN engine used only for shadow evaluation."""

    info = EngineInfo(
        engine_id=ECAPA_ENGINE_ID,
        display_name="ECAPA-TDNN (SpeechBrain)",
        experimental=True,
    )

    def __init__(self, model_cache_directory: str = DEFAULT_MODEL_CACHE_DIR) -> None:
        self._model_cache_directory = Path(model_cache_directory) / ECAPA_ENGINE_ID
        self._classifier: Any = None

    def _load_classifier(self) -> Any:
        """Load/download the pinned public SpeechBrain model on first use."""
        if self._classifier is not None:
            return self._classifier
        try:
            from speechbrain.inference.classifiers import (
                EncoderClassifier,  # type: ignore[import-not-found]
            )
            from speechbrain.utils.fetching import (
                FetchConfig,  # type: ignore[import-not-found]
            )
        except ImportError as error:
            raise RuntimeError(
                "ECAPA shadow evaluation requires the optional SpeechBrain runtime"
            ) from error

        self._model_cache_directory.mkdir(parents=True, exist_ok=True)
        fetch_config = FetchConfig(
            revision=ECAPA_MODEL_REVISION,
            allow_updates=False,
        )
        self._classifier = EncoderClassifier.from_hparams(
            source=ECAPA_MODEL_SOURCE,
            savedir=str(self._model_cache_directory),
            run_opts={"device": "cpu"},
            fetch_config=fetch_config,
        )
        return self._classifier

    def prepare_audio(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Decode mono PCM16 and resample to the ECAPA model's 16 kHz input."""
        audio_bytes = base64.b64decode(audio_input.audio_data)
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).copy()
        if pcm.size == 0:
            raise ValueError("Empty audio data")
        waveform = pcm.astype(np.float32) / 32768.0
        if float(np.max(np.abs(waveform))) < 1e-5:
            raise ValueError("Audio data contains no usable speech signal")
        if audio_input.sample_rate != ECAPA_SAMPLE_RATE:
            common = gcd(audio_input.sample_rate, ECAPA_SAMPLE_RATE)
            waveform = resample_poly(
                waveform,
                ECAPA_SAMPLE_RATE // common,
                audio_input.sample_rate // common,
            ).astype(np.float32, copy=False)
        return waveform.astype(np.float32, copy=False)

    def embed_prepared(self, waveform: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return one flattened ECAPA speaker embedding."""
        classifier = self._load_classifier()
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("ECAPA shadow evaluation requires PyTorch") from error
        tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            embedding = classifier.encode_batch(tensor, normalize=False)
        return np.asarray(embedding.detach().cpu(), dtype=np.float32).reshape(-1)


def available_engines() -> tuple[EngineInfo, ...]:
    """Return metadata for engines known to this service build."""
    return (ResemblyzerEngine.info, EcapaTdnnEngine.info)


def create_engine(
    engine_id: str = DEFAULT_ENGINE_ID,
    *,
    model_cache_directory: str = DEFAULT_MODEL_CACHE_DIR,
) -> SpeakerEmbeddingEngine:
    """Create a speaker embedding engine by stable ID."""
    if engine_id == DEFAULT_ENGINE_ID:
        return ResemblyzerEngine()
    if engine_id == ECAPA_ENGINE_ID:
        return EcapaTdnnEngine(model_cache_directory)
    raise ValueError(f"Unknown speaker embedding engine: {engine_id}")
