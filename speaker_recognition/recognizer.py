"""Speaker recognition logic."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import hashlib
import logging
from pathlib import Path
from time import monotonic
from typing import Any, BinaryIO, Optional

import numpy as np
from numpy.typing import NDArray

from speaker_recognition.const import DEFAULT_ENGINE_ID
from speaker_recognition.engines import SpeakerEmbeddingEngine, create_engine
from speaker_recognition.models import (
    AudioInput,
    Config,
    EnrollmentQualityResult,
    EnrollmentSampleQuality,
    ProfileHealth,
    ProfileHealthResult,
    ProfileSampleSeparation,
    RecognitionRequest,
    RecognitionResult,
    RecognitionScores,
    TrainingRequest,
    TrainingResult,
    config,
)

_LOGGER = logging.getLogger(__name__)
MIN_PROFILE_SAMPLES = 3
MIN_ACCEPTED_SIMILARITY = 0.55
MIN_ACCEPTED_MARGIN = 0.05
PROFILE_SAMPLE_WEIGHT = 0.5
OUTLIER_MIN_GAP = 0.10
PROFILE_SCHEMA_VERSION = 2
LEGACY_PROFILE_SCHEMA_VERSION = 1


class SpeakerRecognizer:
    """Handle speaker recognition operations independently of embedding engine."""

    def __init__(
        self,
        config: Config,
        engine: Optional[SpeakerEmbeddingEngine] = None,
    ) -> None:
        """Initialize the speaker recognizer."""
        self._engine = engine or create_engine()
        self._reference_embeddings: dict[str, NDArray[np.float32]] = {}
        self._sample_embeddings: dict[str, NDArray[np.float32]] = {}
        self._preview_embeddings: OrderedDict[str, tuple[float, NDArray[np.float32]]] = OrderedDict()
        self._is_trained = False
        self._config = config
        self._embeddings_directory = Path(config.embeddings_directory)
        self._load_embeddings()

    @property
    def engine(self) -> SpeakerEmbeddingEngine:
        """Return the active speaker embedding engine."""
        return self._engine

    @property
    def engine_id(self) -> str:
        """Return the stable ID of the active embedding engine."""
        return self._engine.info.engine_id

    @property
    def engine_name(self) -> str:
        """Return the human-readable active embedding engine name."""
        return self._engine.info.display_name

    @property
    def _encoder(self) -> Any:
        """Expose the Resemblyzer encoder for compatibility with older tests/tools."""
        encoder = getattr(self._engine, "encoder", None)
        if encoder is None:
            raise AttributeError("Active embedding engine does not expose an encoder")
        return encoder

    @_encoder.setter
    def _encoder(self, value: Any) -> None:
        if not hasattr(self._engine, "encoder"):
            raise AttributeError("Active embedding engine does not expose an encoder")
        setattr(self._engine, "encoder", value)

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
        """Set the embeddings directory."""
        self._config.embeddings_directory = value
        self._embeddings_directory = Path(value)
        self._load_embeddings()

    def _profile_engine_id(self, profile: Any, schema_version: int) -> str:
        """Return the profile engine, treating schema v1 as Resemblyzer."""
        if schema_version == LEGACY_PROFILE_SCHEMA_VERSION:
            return DEFAULT_ENGINE_ID
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported profile schema version")
        engine_id = str(profile["engine_id"].item())
        if not engine_id:
            raise ValueError("profile contains no embedding engine ID")
        return engine_id

    def _load_embeddings(self) -> None:
        """Load persisted profiles compatible with the active embedding engine."""
        self._reference_embeddings = {}
        self._sample_embeddings = {}
        if not self._embeddings_directory.is_dir():
            self._is_trained = False
            return

        expected_dimension: Optional[int] = None
        for profile_path in self._embeddings_directory.glob("*_profile.npz"):
            try:
                with np.load(profile_path, allow_pickle=False) as profile:
                    schema_version = int(profile["schema_version"].item())
                    if self._profile_engine_id(profile, schema_version) != self.engine_id:
                        continue
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
                    if expected_dimension is not None and embedding.size != expected_dimension:
                        raise ValueError("profile embedding dimension is incompatible")
                    expected_dimension = embedding.size
                    normalized_samples = np.stack(
                        [self._normalize_embedding(sample) for sample in sample_embeddings]
                    ).astype(np.float32, copy=False)
                    self._reference_embeddings[user_id] = embedding
                    self._sample_embeddings[user_id] = normalized_samples
            except (KeyError, OSError, ValueError):
                _LOGGER.warning("Ignoring invalid saved profile: %s", profile_path)

        if self.engine_id == DEFAULT_ENGINE_ID:
            for embedding_path in self._embeddings_directory.glob("*_embedding.npy"):
                user_id = embedding_path.name[: -len("_embedding.npy")]
                if not user_id or user_id in self._reference_embeddings:
                    continue
                try:
                    embedding = self._normalize_embedding(
                        np.load(embedding_path, allow_pickle=False)
                    )
                    if expected_dimension is not None and embedding.size != expected_dimension:
                        raise ValueError("legacy embedding dimension is incompatible")
                    expected_dimension = embedding.size
                    self._reference_embeddings[user_id] = embedding
                    self._sample_embeddings[user_id] = embedding.reshape(1, -1)
                except (OSError, ValueError):
                    _LOGGER.warning("Ignoring invalid saved embedding: %s", embedding_path)

        self._is_trained = bool(self._reference_embeddings)
        if self._is_trained:
            _LOGGER.info(
                "Loaded saved %s embeddings for %d users from %s",
                self.engine_name,
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
        return value / float(np.linalg.norm(value))

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

    def profile_snapshot(self) -> tuple[
        dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]
    ]:
        """Copy immutable diagnostic inputs while the API holds its short lock."""
        return (
            {user: value.copy() for user, value in self._reference_embeddings.items()},
            {user: value.copy() for user, value in self._sample_embeddings.items()
             if isinstance(value, np.ndarray)},
        )

    def profile_health(
        self, snapshot: tuple[dict[str, NDArray[np.float32]], dict[str, NDArray[np.float32]]]
    ) -> ProfileHealthResult:
        """Compute cross-profile diagnostics without inference or derived persistence.

        Distance <0.10 and sample competitor similarity >=0.80 with an own-vs-other
        gap <=0.05 are conservative review heuristics, never recognition gates.
        """
        references, samples = snapshot
        normalized: dict[str, NDArray[np.float32]] = {}
        dimension: Optional[int] = None
        for user in sorted(references):
            try:
                value = self._normalize_embedding(references[user])
                if dimension is not None and value.size != dimension:
                    continue
                dimension = value.size
                normalized[user] = value
            except ValueError:
                continue
        users = list(normalized)
        if not users:
            return ProfileHealthResult(engine_id=self.engine_id, profiles=[])
        centers = np.stack(list(normalized.values()))
        similarities = np.clip(centers @ centers.T, -1.0, 1.0)
        profiles: list[ProfileHealth] = []
        for position, user in enumerate(users):
            valid: list[NDArray[np.float32]] = []
            indexes: list[int] = []
            raw = samples.get(user)
            incomplete = raw is None or raw.ndim != 2 or raw.shape[1] != dimension
            if not incomplete and raw is not None:
                for index, sample in enumerate(raw, 1):
                    try:
                        valid.append(self._normalize_embedding(sample))
                        indexes.append(index)
                    except ValueError:
                        incomplete = True
            consistency = self._profile_diagnostics(np.stack(valid))[0] if len(valid) >= 2 else None
            health = ProfileHealth(
                user_id=user, sample_count=len(valid), internal_consistency=consistency,
                sample_data_incomplete=incomplete or not valid,
            )
            if len(users) > 1:
                competitor = min(
                    (index for index in range(len(users)) if index != position),
                    key=lambda index: (-float(similarities[position, index]), users[index]),
                )
                similarity = float(similarities[position, competitor])
                health.nearest_user_id = users[competitor]
                health.nearest_similarity = similarity
                health.separation = 1.0 - similarity
                health.low_separation = health.separation < 0.10
                for index, sample in zip(indexes, valid):
                    scores = np.clip(centers @ sample, -1.0, 1.0)
                    other = min(
                        (i for i in range(len(users)) if i != position),
                        key=lambda i: (-float(scores[i]), users[i]),
                    )
                    gap = float(scores[position] - scores[other])
                    if float(scores[other]) >= 0.80 and gap <= 0.05:
                        health.sample_warnings.append(ProfileSampleSeparation(
                            sample_index=index, competing_user_id=users[other],
                            competing_similarity=float(scores[other]), separation=gap,
                        ))
            profiles.append(health)
        return ProfileHealthResult(engine_id=self.engine_id, profiles=profiles)

    def _profile_path(self, user_id: str) -> Path:
        """Return a filesystem-safe stable profile path for a user."""
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self._embeddings_directory / f"{digest}_profile.npz"

    def _write_profile(
        self,
        profile_file: BinaryIO,
        user_id: str,
        centroid: NDArray[np.float32],
        sample_embeddings: NDArray[np.float32],
    ) -> None:
        """Write a versioned profile archive with its embedding engine identity."""
        np.savez(
            profile_file,
            schema_version=np.array(PROFILE_SCHEMA_VERSION, dtype=np.int16),
            engine_id=np.array(self.engine_id),
            user_id=np.array(user_id),
            centroid=centroid,
            sample_embeddings=sample_embeddings,
        )

    def _persist_profiles_transactionally(
        self,
        profiles: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    ) -> None:
        """Stage and commit all requested profile files, rolling back on failure."""
        staged: dict[str, Path] = {}
        backups: dict[str, Path] = {}
        committed: list[str] = []
        try:
            for user_id, (centroid, samples) in profiles.items():
                target = self._profile_path(user_id)
                temporary = target.with_suffix(".txn")
                with temporary.open("wb") as profile_file:
                    self._write_profile(profile_file, user_id, centroid, samples)
                staged[user_id] = temporary

            for user_id in profiles:
                target = self._profile_path(user_id)
                backup = target.with_suffix(".bak")
                if target.exists():
                    if backup.exists():
                        backup.unlink()
                    target.replace(backup)
                    backups[user_id] = backup
                staged[user_id].replace(target)
                committed.append(user_id)
        except Exception:
            for user_id in reversed(committed):
                target = self._profile_path(user_id)
                if target.exists():
                    target.unlink()
                restored_backup = backups.get(user_id)
                if restored_backup is not None and restored_backup.exists():
                    restored_backup.replace(target)
            for user_id, backup in backups.items():
                target = self._profile_path(user_id)
                if user_id not in committed and backup.exists() and not target.exists():
                    backup.replace(target)
            raise
        finally:
            for temporary in staged.values():
                if temporary.exists():
                    temporary.unlink()

        for backup in backups.values():
            if backup.exists():
                backup.unlink()
        if self.engine_id == DEFAULT_ENGINE_ID:
            for user_id in profiles:
                legacy_name = f"{user_id}_embedding.npy"
                if Path(legacy_name).name == legacy_name:
                    legacy = self._embeddings_directory / legacy_name
                    if legacy.exists():
                        legacy.unlink()

    def sync_profiles(self, desired_users: set[str]) -> list[str]:
        """Remove persisted profiles that are no longer configured."""
        removed: list[str] = []
        for user_id in sorted(set(self._reference_embeddings) - desired_users):
            path = self._profile_path(user_id)
            if path.exists():
                path.unlink()
            if self.engine_id == DEFAULT_ENGINE_ID:
                legacy_name = f"{user_id}_embedding.npy"
                if Path(legacy_name).name == legacy_name:
                    legacy = self._embeddings_directory / legacy_name
                    if legacy.exists():
                        legacy.unlink()
            self._reference_embeddings.pop(user_id, None)
            self._sample_embeddings.pop(user_id, None)
            removed.append(user_id)
        self._is_trained = bool(self._reference_embeddings)
        return removed

    def process_audio_input(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Prepare audio using the active embedding engine."""
        return self._engine.prepare_audio(audio_input)

    def _embed_audio(self, audio_input: AudioInput) -> NDArray[np.float32]:
        """Return a validated embedding from the active engine."""
        waveform = self.process_audio_input(audio_input)
        return self._validate_embedding(
            np.asarray(self._engine.embed_prepared(waveform), dtype=np.float32)
        )

    def _preview_key(self, audio: AudioInput) -> str:
        return hashlib.sha256(
            f"{self.engine_id}:{audio.sample_rate}:{audio.audio_data}".encode("ascii")
        ).hexdigest()

    def _staged_embedding(self, audio: AudioInput, *, preview: bool = False) -> NDArray[np.float32]:
        """Reuse only successful preview embeddings, bounded to 48 for 15 minutes.

        Recognition never consults this cache. Normal training without a preview
        follows the original inference path, including repeated recordings.
        """
        key = self._preview_key(audio)
        now = monotonic()
        for expired in [key for key, (created, _) in self._preview_embeddings.items() if now - created > 900]:
            self._preview_embeddings.pop(expired)
        cached = self._preview_embeddings.get(key)
        if cached is not None:
            self._preview_embeddings.move_to_end(key)
            return cached[1].copy()
        embedding = self._embed_audio(audio)
        if preview:
            self._preview_embeddings[key] = (now, embedding.copy())
            while len(self._preview_embeddings) > 48:
                self._preview_embeddings.popitem(last=False)
        return embedding

    def enrollment_quality(self, request: TrainingRequest) -> EnrollmentQualityResult:
        """Assess staged recordings without changing or persisting any profile."""
        users = {sample.user for sample in request.voice_samples}
        if len(users) != 1 or len(request.voice_samples) > 12:
            raise ValueError("Quality preview requires one user and at most twelve staged samples")
        user_id = next(iter(users))
        embeddings = np.stack([
            self._normalize_embedding(self._staged_embedding(sample.audio, preview=True))
            for sample in request.voice_samples
        ]).astype(np.float32, copy=False)
        if self._reference_embeddings and embeddings.shape[1] != next(iter(self._reference_embeddings.values())).size:
            raise ValueError("Sample embedding dimensions do not match profiles")
        enough = len(embeddings) >= MIN_PROFILE_SAMPLES
        consistency, outliers = self._profile_diagnostics(embeddings)
        reference = self._reference_embeddings.get(user_id)
        return EnrollmentQualityResult(
            engine_id=self.engine_id,
            consistency=consistency if enough else None,
            samples=[EnrollmentSampleQuality(
                sample_index=index + 1,
                assessment=("insufficient_evidence" if not enough else
                            "inconsistent" if index + 1 in outliers or consistency < 0.5 else "good"),
                outlier=enough and index + 1 in outliers,
                profile_similarity=float(np.dot(embedding, reference)) if reference is not None else None,
            ) for index, embedding in enumerate(embeddings)],
        )

    def train(self, request: TrainingRequest) -> TrainingResult:
        """Build all requested profiles first, then commit them as one transaction."""
        if not request.voice_samples:
            raise ValueError("No voice samples provided")
        self._embeddings_directory.mkdir(parents=True, exist_ok=True)
        samples_by_user = defaultdict(list)
        for sample in request.voice_samples:
            samples_by_user[sample.user].append(sample.audio)

        accepted_samples: dict[str, int] = {}
        rejected_samples: dict[str, int] = {}
        profile_consistency: dict[str, float] = {}
        outlier_samples: dict[str, list[int]] = {}
        proposed: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
        existing_dimension = (
            next(iter(self._reference_embeddings.values())).size
            if self._reference_embeddings
            else None
        )
        request_dimension: Optional[int] = existing_dimension

        for user_id, audio_inputs in samples_by_user.items():
            embeddings: list[NDArray[np.float32]] = []
            for sample_number, audio_input in enumerate(audio_inputs, start=1):
                try:
                    embedding = self._staged_embedding(audio_input)
                    if request_dimension is not None and embedding.size != request_dimension:
                        raise ValueError("sample embedding dimensions do not match profiles")
                    request_dimension = embedding.size
                    embeddings.append(embedding)
                except Exception as error:
                    _LOGGER.error(
                        "Error processing voice sample %d for user %s: %s",
                        sample_number,
                        user_id,
                        error,
                    )

            if embeddings:
                raw_embeddings = np.stack(embeddings).astype(np.float32, copy=False)
                consistency, outliers = self._profile_diagnostics(raw_embeddings)
            else:
                raw_embeddings = np.empty((0, request_dimension or 0), dtype=np.float32)
                consistency, outliers = 0.0, []

            outlier_indexes = {index - 1 for index in outliers}
            retained = [
                embedding
                for index, embedding in enumerate(embeddings)
                if index not in outlier_indexes
            ]
            accepted_samples[user_id] = len(retained)
            rejected_samples[user_id] = len(audio_inputs) - len(retained)
            profile_consistency[user_id] = consistency
            outlier_samples[user_id] = outliers

            if len(retained) < MIN_PROFILE_SAMPLES:
                raise ValueError(
                    f"User {user_id} has only {len(retained)} usable non-outlier samples; "
                    f"at least {MIN_PROFILE_SAMPLES} are required"
                )

            sample_embeddings = np.stack(retained).astype(np.float32, copy=False)
            normalized_samples = np.stack(
                [self._normalize_embedding(sample) for sample in sample_embeddings]
            ).astype(np.float32, copy=False)
            centroid = self._normalize_embedding(normalized_samples.mean(axis=0))
            proposed[user_id] = (centroid, normalized_samples)

        self._persist_profiles_transactionally(proposed)
        for user_id, (centroid, normalized_samples) in proposed.items():
            self._reference_embeddings[user_id] = centroid
            self._sample_embeddings[user_id] = normalized_samples
        self._is_trained = bool(self._reference_embeddings)

        updated_users = sorted(proposed)
        return TrainingResult(
            status="success",
            trained_users=updated_users,
            count=len(self._reference_embeddings),
            engine_id=self.engine_id,
            accepted_samples=accepted_samples,
            rejected_samples=rejected_samples,
            profile_consistency=profile_consistency,
            outlier_samples=outlier_samples,
        )

    def score(self, request: RecognitionRequest) -> RecognitionScores:
        """Return raw speaker scores before applying acceptance thresholds."""
        if not self._is_trained or not self._reference_embeddings:
            raise RuntimeError("Model not trained")
        chunk_embedding = self._normalize_embedding(self._embed_audio(request.audio))
        expected_dimension = next(iter(self._reference_embeddings.values())).size
        if chunk_embedding.size != expected_dimension:
            raise ValueError("Recognition embedding dimension does not match profiles")

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
        return RecognitionScores(
            engine_id=self.engine_id,
            candidate_user_id=candidate_user_id,
            similarity=best_score,
            margin=margin,
            all_scores=scores,
        )

    def recognize(self, request: RecognitionRequest) -> RecognitionResult:
        """Recognize or reject a speaker from audio data."""
        scores = self.score(request)
        policy = request.acceptance_thresholds or self._config.acceptance_thresholds
        accepted = scores.similarity >= policy.min_similarity and (
            scores.margin is None or scores.margin >= policy.min_margin
        )
        return RecognitionResult(
            acceptance_thresholds=policy,
            engine_id=scores.engine_id,
            user_id=scores.candidate_user_id if accepted else None,
            candidate_user_id=scores.candidate_user_id,
            confidence=scores.similarity,
            similarity=scores.similarity,
            margin=scores.margin,
            accepted=accepted,
            all_scores=scores.all_scores,
        )


recognizer = SpeakerRecognizer(config=config)
