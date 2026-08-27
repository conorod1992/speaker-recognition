"""Speaker recognition logic."""

import base64
from collections import defaultdict
import hashlib
import logging
from pathlib import Path
from typing import BinaryIO, Optional

import numpy as np
from numpy.typing import NDArray
from resemblyzer import VoiceEncoder, preprocess_wav  # type: ignore[import-untyped]

from speaker_recognition.models import (
    AudioInput,
    Config,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    config,
)

_LOGGER = logging.getLogger(__name__)
MIN_RETRAINING_SAMPLES = 3
MIN_ACCEPTED_SIMILARITY = 0.55
MIN_ACCEPTED_MARGIN = 0.05
PROFILE_SAMPLE_WEIGHT = 0.5
OUTLIER_MIN_GAP = 0.10


class SpeakerRecognizer:
    """Handle speaker recognition operations."""

    def __init__(self, config: Config) -> None:
        """Initialize the speaker recognizer.

        Args:
            config: Application configuration
        """
        self._encoder: VoiceEncoder = VoiceEncoder()
        self._reference_embeddings: dict[str, NDArray[np.float32]] = {}
        self._sample_embeddings: dict[str, NDArray[np.float32]] = {}
        self._is_trained = False
        self._config = config
        self._embeddings_directory = Path(config.embeddings_directory)
        self._load_embeddings()

    @property
    def is_trained(self) -> bool:
        """Check if the model is trained."""
        return self._is_trained

    @property
    def enrolled_users(self) -> list[str]:
        """Return user IDs with usable persisted reference profiles."""
        return sorted(self._reference_embeddings)

    @property
    def embeddings_directory(self) -> Path:
        """Get the embeddings directory."""
        return self._embeddings_directory

    @embeddings_directory.setter
    def embeddings_directory(self, value: str) -> None:
        """Set the embeddings directory.

        Args:
            value: New embeddings directory path
        """
        self._config.embeddings_directory = value
        self._embeddings_directory = Path(value)
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        """Load persisted profiles and legacy single-embedding files."""
        self._reference_embeddings = {}
        self._sample_embeddings = {}
        if not self._embeddings_directory.is_dir():
            self._is_trained = False
            return

        for profile_path in self._embeddings_directory.glob("*_profile.npz"):
            try:
                with np.load(profile_path, allow_pickle=False) as profile:
                    user_id = str(profile["user_id"].item())
                    embedding = self._normalize_embedding(profile["centroid"])
                    sample_embeddings = np.asarray(profile["sample_embeddings"])
                    if (
                        not user_id
                        or sample_embeddings.ndim != 2
                        or sample_embeddings.shape[0] == 0
                        or sample_embeddings.shape[1] != embedding.size
                        or not np.isfinite(sample_embeddings).all()
                    ):
                        raise ValueError("profile contains invalid sample embeddings")
                    normalized_samples = np.stack(
                        [self._normalize_embedding(sample) for sample in sample_embeddings]
                    ).astype(np.float32, copy=False)
                    self._reference_embeddings[user_id] = embedding
                    self._sample_embeddings[user_id] = normalized_samples
            except (KeyError, OSError, ValueError):
                _LOGGER.warning("Ignoring invalid saved profile: %s", profile_path)

        # Version 1 stored one ``<user>_embedding.npy`` file per user. Continue
        # loading those profiles until that user is enrolled again.
        for embedding_path in self._embeddings_directory.glob("*_embedding.npy"):
            user_id = embedding_path.name[: -len("_embedding.npy")]
            if not user_id or user_id in self._reference_embeddings:
                continue
            try:
                embedding = self._normalize_embedding(
                    np.load(embedding_path, allow_pickle=False)
                )
                self._reference_embeddings[user_id] = embedding
                self._sample_embeddings[user_id] = embedding.reshape(1, -1)
            except (OSError, ValueError):
                _LOGGER.warning("Ignoring invalid saved embedding: %s", embedding_path)

        self._is_trained = bool(self._reference_embeddings)
        if self._is_trained:
            _LOGGER.info(
                "Loaded saved embeddings for %d users from %s",
                len(self._reference_embeddings),
                self._embeddings_directory,
            )

    @staticmethod
    def _validate_embedding(embedding: NDArray[np.generic]) -> NDArray[np.float32]:
        """Return an embedding after rejecting invalid encoder output."""
        value = np.asarray(embedding, dtype=np.float32)
        if value.ndim != 1 or value.size == 0 or not np.isfinite(value).all():
            raise ValueError("embedding must be a finite, one-dimensional array")
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("embedding must have a non-zero norm")
        return value

    @classmethod
    def _normalize_embedding(
        cls, embedding: NDArray[np.generic]
    ) -> NDArray[np.float32]:
        """Validate and L2-normalize an embedding."""
        value = cls._validate_embedding(embedding)
        norm = float(np.linalg.norm(value))
        return value / norm

    @classmethod
    def _profile_diagnostics(
        cls, sample_embeddings: NDArray[np.float32]
    ) -> tuple[float, list[int]]:
        """Return enrollment consistency and one-based outlier sample indexes."""
        normalized = np.stack(
            [cls._normalize_embedding(sample) for sample in sample_embeddings]
        ).astype(np.float32, copy=False)
        sample_count = normalized.shape[0]
        if sample_count < 2:
            return 1.0, []

        similarities = normalized @ normalized.T
        upper = similarities[np.triu_indices(sample_count, k=1)]
        consistency = float(np.mean(upper))

        peer_scores = (similarities.sum(axis=1) - 1.0) / (sample_count - 1)
        median = float(np.median(peer_scores))
        mad = float(np.median(np.abs(peer_scores - median)))
        cutoff = median - max(OUTLIER_MIN_GAP, 2.0 * mad)
        outliers = [
            index + 1
            for index, score in enumerate(peer_scores)
            if float(score) < cutoff
        ]
        return consistency, outliers

    @staticmethod
    def _profile_score(
        centroid: NDArray[np.float32],
        sample_embeddings: NDArray[np.float32],
        chunk_embedding: NDArray[np.float32],
    ) -> float:
        """Combine centroid similarity with the strongest enrollment examples."""
        centroid_score = float(np.dot(centroid, chunk_embedding))
        if sample_embeddings.shape[0] < 2:
            return centroid_score

        sample_scores = sample_embeddings @ chunk_embedding
        strongest_count = min(2, sample_scores.size)
        strongest = np.partition(sample_scores, -strongest_count)[-strongest_count:]
        sample_score = float(np.mean(strongest))
        return (
            (1.0 - PROFILE_SAMPLE_WEIGHT) * centroid_score
            + PROFILE_SAMPLE_WEIGHT * sample_score
        )

    def _profile_path(self, user_id: str) -> Path:
        """Return a filesystem-safe stable profile path for a user."""
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self._embeddings_directory / f"{digest}_profile.npz"

    def _save_profile(
        self,
        user_id: str,
        centroid: NDArray[np.float32],
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Atomically persist a reference and its calibration-ready samples."""
        profile_path = self._profile_path(user_id)
        temporary_path = profile_path.with_suffix(".tmp")
        with temporary_path.open("wb") as profile_file:
            self._write_profile(profile_file, user_id, centroid, sample_embeddings)
        temporary_path.replace(profile_path)

        # Remove the legacy reference for this user after the new profile is safe.
        legacy_name = f"{user_id}_embedding.npy"
        if Path(legacy_name).name == legacy_name:
            legacy_path = self._embeddings_directory / legacy_name
            if legacy_path.exists():
                legacy_path.unlink()

    @staticmethod
    def _write_profile(
        profile_file: BinaryIO,
        user_id: str,
        centroid: NDArray[np.float32],
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Write a versioned profile archive to an open file."""
        np.savez(
            profile_file,
            schema_version=np.array(1, dtype=np.int16),
            user_id=np.array(user_id),
            centroid=centroid,
            sample_embeddings=sample_embeddings,
        )

    def process_audio_input(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Process audio input from base64 encoded data.

        Args:
            audio_input: Audio input containing base64 encoded audio

        Returns:
            Preprocessed audio waveform
        """
        audio_bytes = base64.b64decode(audio_input.audio_data)
        audio_array_int16 = np.frombuffer(audio_bytes, dtype=np.int16).copy()

        if audio_array_int16.size == 0:
            raise ValueError("Empty audio data")

        audio_array_float32 = audio_array_int16.astype(np.float32) / 32768.0
        if float(np.max(np.abs(audio_array_float32))) < 1e-5:
            raise ValueError("Audio data contains no usable speech signal")
        result: NDArray[np.float32] = preprocess_wav(
            audio_array_float32, source_sr=audio_input.sample_rate
        )
        return result

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Train the speaker recognition model.

        Args:
            request: Training request with voice samples

        Returns:
            TrainingResult with status, trained users and count
        """
        if not request.voice_samples:
            raise ValueError("No voice samples provided")

        self._embeddings_directory.mkdir(parents=True, exist_ok=True)

        _LOGGER.info(f"Training with {len(request.voice_samples)} voice samples")
        samples_by_user = defaultdict(list)
        for sample in request.voice_samples:
            samples_by_user[sample.user].append(sample.audio)

        accepted_samples: dict[str, int] = {}
        rejected_samples: dict[str, int] = {}
        profile_consistency: dict[str, float] = {}
        outlier_samples: dict[str, list[int]] = {}
        updated_users: list[str] = []

        for user_id, audio_inputs in samples_by_user.items():
            embeddings: list[NDArray[np.float32]] = []
            expected_size: Optional[int] = None
            for sample_number, audio_input in enumerate(audio_inputs, start=1):
                try:
                    _LOGGER.debug(
                        "Creating embedding for user %s sample %d",
                        user_id,
                        sample_number,
                    )
                    wav = self.process_audio_input(audio_input)
                    embedding = self._validate_embedding(
                        np.asarray(self._encoder.embed_utterance(wav))
                    )
                    if expected_size is not None and embedding.size != expected_size:
                        raise ValueError("sample embedding dimensions do not match")
                    expected_size = embedding.size
                    embeddings.append(embedding)
                except Exception as error:
                    _LOGGER.error(
                        "Error processing voice sample %d for user %s: %s",
                        sample_number,
                        user_id,
                        error,
                    )

            accepted_samples[user_id] = len(embeddings)
            rejected_samples[user_id] = len(audio_inputs) - len(embeddings)
            if not embeddings:
                continue

            if (
                user_id in self._reference_embeddings
                and len(embeddings) < MIN_RETRAINING_SAMPLES
            ):
                _LOGGER.warning(
                    "Refusing to replace existing profile for %s with only %d "
                    "accepted samples; at least %d are required",
                    user_id,
                    len(embeddings),
                    MIN_RETRAINING_SAMPLES,
                )
                continue

            sample_embeddings = np.stack(embeddings).astype(np.float32, copy=False)
            normalized_samples = np.stack(
                [self._normalize_embedding(sample) for sample in sample_embeddings]
            ).astype(np.float32, copy=False)
            centroid = self._normalize_embedding(sample_embeddings.mean(axis=0))
            consistency, outliers = self._profile_diagnostics(sample_embeddings)
            self._save_profile(user_id, centroid, sample_embeddings)
            self._reference_embeddings[user_id] = centroid
            self._sample_embeddings[user_id] = normalized_samples
            profile_consistency[user_id] = consistency
            outlier_samples[user_id] = outliers
            updated_users.append(user_id)
            _LOGGER.info(
                "Enrolled user %s with %d accepted samples (%d rejected), "
                "consistency %.3f, outliers %s",
                user_id,
                accepted_samples[user_id],
                rejected_samples[user_id],
                consistency,
                outliers,
            )

        if updated_users:
            self._is_trained = True
            _LOGGER.info(
                "Training completed for %d users; %d total profiles available",
                len(updated_users),
                len(self._reference_embeddings),
            )
            return TrainingResult(
                status="success",
                trained_users=updated_users,
                count=len(self._reference_embeddings),
                accepted_samples=accepted_samples,
                rejected_samples=rejected_samples,
                profile_consistency=profile_consistency,
                outlier_samples=outlier_samples,
            )

        self._is_trained = bool(self._reference_embeddings)
        raise ValueError("No valid voice profiles updated")

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize or reject a speaker from audio data."""
        if not self._is_trained or not self._reference_embeddings:
            raise RuntimeError("Model not trained")

        wav = self.process_audio_input(request.audio)
        chunk_embedding = self._normalize_embedding(
            np.asarray(self._encoder.embed_utterance(wav))
        )

        scores: dict[str, float] = {}
        for user_id, reference_embedding in self._reference_embeddings.items():
            samples = self._sample_embeddings.get(user_id)
            if samples is None:
                samples = reference_embedding.reshape(1, -1)
            scores[user_id] = self._profile_score(
                reference_embedding, samples, chunk_embedding
            )

        if not scores:
            raise RuntimeError("No scores calculated")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        candidate_user_id, best_score = ranked[0]
        margin = best_score - ranked[1][1] if len(ranked) > 1 else None
        accepted = best_score >= MIN_ACCEPTED_SIMILARITY and (
            margin is None or margin >= MIN_ACCEPTED_MARGIN
        )
        recognized_user_id = candidate_user_id if accepted else None

        _LOGGER.debug(
            "Recognition candidate=%s similarity=%.3f margin=%s accepted=%s scores=%s",
            candidate_user_id,
            best_score,
            f"{margin:.3f}" if margin is not None else "n/a",
            accepted,
            scores,
        )

        return RecognitionResult(
            user_id=recognized_user_id,
            candidate_user_id=candidate_user_id,
            confidence=best_score,
            similarity=best_score,
            margin=margin,
            accepted=accepted,
            all_scores=scores,
        )


recognizer = SpeakerRecognizer(config=config)
