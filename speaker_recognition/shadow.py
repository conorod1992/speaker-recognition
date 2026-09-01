"""Optional non-authoritative speaker-recognition shadow runtime."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from speaker_recognition.const import DEFAULT_SHADOW_ENGINE
from speaker_recognition.engines import create_engine
from speaker_recognition.models import Config
from speaker_recognition.recognizer import SpeakerRecognizer

_LOGGER = logging.getLogger(__name__)


class ShadowRecognitionService:
    """Own a separately persisted experimental recognizer when configured."""

    def __init__(self) -> None:
        self._recognizer: Optional[SpeakerRecognizer] = None
        self._last_error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._recognizer is not None

    @property
    def recognizer(self) -> SpeakerRecognizer:
        if self._recognizer is None:
            raise RuntimeError("No shadow speaker engine is configured")
        return self._recognizer

    @property
    def engine_id(self) -> Optional[str]:
        return self._recognizer.engine_id if self._recognizer is not None else None

    @property
    def engine_name(self) -> Optional[str]:
        return self._recognizer.engine_name if self._recognizer is not None else None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def clear_error(self) -> None:
        self._last_error = None

    def record_error(self, error: Exception) -> None:
        self._last_error = f"{type(error).__name__}: {error}"

    def configure(self, config: Config) -> None:
        """Configure a lazy shadow engine without warming it on startup."""
        engine_id = config.shadow_engine.strip().lower()
        if not engine_id or engine_id == DEFAULT_SHADOW_ENGINE:
            self._recognizer = None
            self._last_error = None
            _LOGGER.info("Experimental speaker shadow engine is disabled")
            return

        engine = create_engine(
            engine_id,
            model_cache_directory=config.model_cache_directory,
        )
        shadow_directory = (
            Path(config.embeddings_directory) / "shadow" / engine.info.engine_id
        )
        shadow_config = Config(
            embeddings_directory=str(shadow_directory),
            model_cache_directory=config.model_cache_directory,
        )
        self._recognizer = SpeakerRecognizer(config=shadow_config, engine=engine)
        self._last_error = None
        _LOGGER.info(
            "Configured experimental shadow engine %s with %d persisted profiles",
            self._recognizer.engine_name,
            len(self._recognizer.enrolled_users),
        )


shadow_service = ShadowRecognitionService()
