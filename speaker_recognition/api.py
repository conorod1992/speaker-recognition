"""FastAPI application for speaker recognition service."""

from __future__ import annotations

import base64
import binascii
from ipaddress import ip_address
import logging
import secrets
from threading import Lock
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from speaker_recognition.models import (
    ErrorResponse,
    HealthResponse,
    ProfileSyncRequest,
    ProfileSyncResult,
    RecognitionRequest,
    RecognitionResult,
    TrainingRequest,
    TrainingResult,
    config,
)
from speaker_recognition.recognizer import recognizer
from speaker_recognition.warmup import WarmupStatus, warm_encoder

_LOGGER = logging.getLogger(__name__)
_RECOGNIZER_LOCK = Lock()
_WARMUP_STATUS: WarmupStatus = warm_encoder(recognizer)
_TRUSTED_LOCAL_HOSTS = {"172.30.32.1"}


def _is_trusted_local(host: Optional[str]) -> bool:
    """Return whether a request source is local to the service/HA host."""
    if not host:
        return False
    if host in _TRUSTED_LOCAL_HOSTS:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _authorization_token(authorization: Optional[str]) -> str:
    """Extract a bearer token or a token supplied via HTTP Basic credentials."""
    if not authorization:
        return ""
    if authorization.startswith("Bearer "):
        return authorization[7:]
    if not authorization.startswith("Basic "):
        return ""
    try:
        decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError):
        return ""
    username, separator, password = decoded.partition(":")
    if not separator:
        return ""
    return username or password


async def require_api_access(
    request: Request,
    authorization: Optional[str] = Header(default=None),
) -> None:
    """Allow trusted local callers, or require the configured token remotely."""
    client_host = request.client.host if request.client is not None else None
    if _is_trusted_local(client_host) or config.allow_insecure_remote:
        return
    expected = config.api_token.strip()
    supplied = _authorization_token(authorization)
    if expected and supplied and secrets.compare_digest(supplied, expected):
        return
    raise HTTPException(
        status_code=401,
        detail="Speaker Recognition API authentication required",
    )


app = FastAPI(
    title="Speaker Recognition Service",
    description="API for training and recognizing speakers using voice samples",
    version="2.8.0",
    dependencies=[Depends(require_api_access)],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint, retrying a failed startup warm-up."""
    global _WARMUP_STATUS
    with _RECOGNIZER_LOCK:
        if not _WARMUP_STATUS.ready:
            _WARMUP_STATUS = warm_encoder(recognizer)
        return HealthResponse(
            status="healthy" if _WARMUP_STATUS.ready else "degraded",
            trained=recognizer.is_trained,
            enrolled_users=recognizer.enrolled_users,
            encoder_ready=_WARMUP_STATUS.ready,
            warmup_seconds=_WARMUP_STATUS.seconds,
            warmup_error=_WARMUP_STATUS.error,
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
        _LOGGER.error("Validation error during training: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _LOGGER.exception("Error during training")
        raise HTTPException(
            status_code=500,
            detail="Speaker recognition training failed",
        ) from error


@app.post(
    "/profiles/sync",
    response_model=ProfileSyncResult,
    responses={500: {"model": ErrorResponse}},
    tags=["Training"],
)
def sync_profiles(request: ProfileSyncRequest) -> ProfileSyncResult:
    """Remove persisted profiles no longer present in HA configuration."""
    try:
        with _RECOGNIZER_LOCK:
            removed_users = recognizer.sync_profiles(set(request.desired_users))
            return ProfileSyncResult(
                enrolled_users=recognizer.enrolled_users,
                removed_users=removed_users,
            )
    except Exception as error:
        _LOGGER.exception("Error synchronizing speaker profiles")
        raise HTTPException(
            status_code=500,
            detail="Speaker profile synchronization failed",
        ) from error


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
        _LOGGER.error("Recognition error: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _LOGGER.exception("Error during recognition")
        raise HTTPException(
            status_code=500,
            detail="Speaker recognition failed",
        ) from error
