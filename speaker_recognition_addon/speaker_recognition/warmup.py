"""Startup warm-up for speaker embedding engines."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from time import perf_counter
from typing import Any, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)
_WARMUP_SAMPLE_RATE = 16000
_WARMUP_SECONDS = 0.8


@dataclass(frozen=True)
class WarmupStatus:
    """Result of the one-time embedding-engine warm-up."""

    ready: bool
    seconds: float
    error: Optional[str] = None


def warm_engine(engine: Any) -> WarmupStatus:
    """Run one deterministic embedding before the API starts accepting traffic."""
    if os.environ.get("SPEAKER_RECOGNITION_SKIP_WARMUP") == "1":
        _LOGGER.debug("Skipping speaker encoder warm-up by environment request")
        return WarmupStatus(ready=True, seconds=0.0)

    started = perf_counter()
    try:
        sample_count = int(_WARMUP_SAMPLE_RATE * _WARMUP_SECONDS)
        time_axis = np.arange(sample_count, dtype=np.float32) / _WARMUP_SAMPLE_RATE
        envelope = np.sin(np.pi * np.arange(sample_count, dtype=np.float32) / sample_count)
        waveform = (
            0.08 * np.sin(2.0 * math.pi * 180.0 * time_axis)
            + 0.04 * np.sin(2.0 * math.pi * 360.0 * time_axis)
            + 0.02 * np.sin(2.0 * math.pi * 720.0 * time_axis)
        ) * envelope
        if hasattr(engine, "embed_prepared"):
            raw_embedding = engine.embed_prepared(waveform)
        else:
            encoder = getattr(engine, "_encoder", None)
            if encoder is None:
                raise AttributeError("embedding engine does not expose an embedding method")
            raw_embedding = encoder.embed_utterance(waveform)
        embedding = np.asarray(raw_embedding, dtype=np.float32)
        if embedding.ndim != 1 or embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError("encoder returned an invalid warm-up embedding")
    except Exception as error:  # Warm-up failure must not make the service unusable.
        elapsed = perf_counter() - started
        detail = f"{type(error).__name__}: {error}"
        _LOGGER.exception("Speaker encoder warm-up failed after %.3fs", elapsed)
        return WarmupStatus(ready=False, seconds=elapsed, error=detail)

    elapsed = perf_counter() - started
    _LOGGER.info("Speaker encoder warm-up completed in %.3fs", elapsed)
    return WarmupStatus(ready=True, seconds=elapsed)


def warm_encoder(recognizer: Any) -> WarmupStatus:
    """Compatibility wrapper accepting a recognizer or legacy encoder holder."""
    return warm_engine(getattr(recognizer, "engine", recognizer))
