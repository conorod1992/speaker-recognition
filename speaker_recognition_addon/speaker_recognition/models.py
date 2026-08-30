"""Data models for speaker recognition."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from speaker_recognition.const import (
    DEFAULT_ACCESS_LOG,
    DEFAULT_ALLOW_INSECURE_REMOTE,
    DEFAULT_API_TOKEN,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    MAX_AUDIO_BASE64_CHARS,
    MAX_SAMPLE_RATE,
    MAX_TRAINING_AUDIO_BYTES,
    MAX_TRAINING_SAMPLES,
    MIN_SAMPLE_RATE,
)


class Config(BaseModel):
    """Application configuration."""

    model_config = ConfigDict(validate_assignment=True)

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    log_level: str = DEFAULT_LOG_LEVEL
    access_log: bool = DEFAULT_ACCESS_LOG
    embeddings_directory: str = DEFAULT_EMBEDDINGS_DIR
    api_token: str = DEFAULT_API_TOKEN
    allow_insecure_remote: bool = DEFAULT_ALLOW_INSECURE_REMOTE


class AudioInput(BaseModel):
    """Audio input data model."""

    audio_data: str = Field(
        ...,
        min_length=1,
        max_length=MAX_AUDIO_BASE64_CHARS,
        description="Base64 encoded audio data",
    )
    sample_rate: int = Field(
        16000,
        ge=MIN_SAMPLE_RATE,
        le=MAX_SAMPLE_RATE,
        description="Audio sample rate in Hz",
    )


class VoiceSample(BaseModel):
    """Audio sample associated with one user."""

    user: str = Field(..., min_length=1, max_length=256, description="User identifier")
    audio: AudioInput = Field(..., description="Audio input for voice sample")


class TrainingRequest(BaseModel):
    """Training request data model."""

    voice_samples: list[VoiceSample] = Field(
        ...,
        min_length=1,
        max_length=MAX_TRAINING_SAMPLES,
        description="List of voice samples",
    )

    @model_validator(mode="after")
    def validate_total_audio_budget(self) -> "TrainingRequest":
        """Reject requests whose combined decoded audio can exhaust memory."""
        estimated_bytes = sum(
            (len(sample.audio.audio_data) * 3 + 3) // 4
            for sample in self.voice_samples
        )
        if estimated_bytes > MAX_TRAINING_AUDIO_BYTES:
            raise ValueError(
                "Combined training audio exceeds the 64 MiB request budget"
            )
        return self


class TrainingResult(BaseModel):
    """Result of training operation."""

    status: str
    trained_users: list[str]
    count: int
    accepted_samples: dict[str, int] = Field(default_factory=dict)
    rejected_samples: dict[str, int] = Field(default_factory=dict)
    profile_consistency: dict[str, float] = Field(default_factory=dict)
    outlier_samples: dict[str, list[int]] = Field(default_factory=dict)


class ProfileSyncRequest(BaseModel):
    """Desired set of persisted speaker profiles."""

    desired_users: list[str] = Field(default_factory=list, max_length=256)


class ProfileSyncResult(BaseModel):
    """Result of synchronizing persisted profiles."""

    enrolled_users: list[str]
    removed_users: list[str] = Field(default_factory=list)


class RecognitionRequest(BaseModel):
    """Recognition request data model."""

    audio: AudioInput = Field(..., description="Audio input for recognition")


class RecognitionResult(BaseModel):
    """Result of recognition operation."""

    user_id: Optional[str]
    candidate_user_id: str
    confidence: float
    similarity: float
    margin: Optional[float] = None
    accepted: bool
    all_scores: dict[str, float]


class DenoiseRequest(BaseModel):
    """Neural speech denoise request."""

    audio: AudioInput = Field(..., description="Mono PCM16 audio to denoise")


class DenoiseResult(BaseModel):
    """Neural speech denoise response."""

    audio_data: str = Field(..., description="Base64 encoded denoised mono PCM16")
    sample_rate: int
    processing_seconds: float
    engine: str = "rnnoise"


class HealthResponse(BaseModel):
    """Health check response data model."""

    status: str
    trained: bool = False
    enrolled_users: list[str] = Field(default_factory=list)
    encoder_ready: bool = False
    warmup_seconds: Optional[float] = None
    warmup_error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response data model."""

    error: str


config = Config()
