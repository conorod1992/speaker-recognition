"""Startup warm-up for the speaker embedding encoder."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from time import perf_counter
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)
_WARMUP_SAMPLE_RATE = 16000
_WARMUP_SECONDS = 0.8


@dataclass(frozen=True)
class WarmupStatus:
    """Result of the one-time encoder warm-up."""

    ready: bool
    seconds: float
    error: str | None = None


def warm_encoder(recognizer: Any) -> WarmupStatus:
    """Run one deterministic embedding before the API starts accepting traffic."""
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
        embedding = np.asarray(recognizer._encoder.embed_utterance(waveform), dtype=np.float32)
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
