"""FastAPI application for speaker recognition service."""

import logging
from threading import Lock

from fastapi import FastAPI, HTTPException

from speaker_recognition.models import (
    ErrorResponse,
    HealthResponse,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
)
from speaker_recognition.recognizer import recognizer

_LOGGER = logging.getLogger(__name__)
_RECOGNIZER_LOCK = Lock()

app = FastAPI(
    title="Speaker Recognition Service",
    description="API for training and recognizing speakers using voice samples",
    version="2.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint."""
    with _RECOGNIZER_LOCK:
        return HealthResponse(
            status="healthy",
            trained=recognizer.is_trained,
            enrolled_users=recognizer.enrolled_users,
        )


@app.post(
    "/train",
    response_model=TrainingResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Training"],
)
def train(request: TrainingRequest) -> TrainingResult:
    """Train the speaker recognition model."""
    try:
        with _RECOGNIZER_LOCK:
            return recognizer.train(request)

    except ValueError as error:
        _LOGGER.error(f"Validation error during training: {error}")
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        _LOGGER.error(f"Error during training: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post(
    "/recognize",
    response_model=RecognitionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Recognition"],
)
def recognize(request: RecognitionRequest) -> RecognitionResult:
    """Recognize speaker from audio data."""
    try:
        with _RECOGNIZER_LOCK:
            return recognizer.recognize(request)

    except (ValueError, RuntimeError) as error:
        _LOGGER.error(f"Recognition error: {error}")
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        _LOGGER.error(f"Error during recognition: {error}")
        raise HTTPException(status_code=500, detail=str(error))
