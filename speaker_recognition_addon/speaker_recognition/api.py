"""FastAPI application for speaker recognition service."""

from __future__ import annotations

import base64
import binascii
from ipaddress import ip_address
import logging
import os
import secrets
from threading import Lock
from time import perf_counter
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from speaker_recognition.const import MAX_REQUEST_BODY_BYTES
from speaker_recognition.models import (
    ErrorResponse,
    HealthResponse,
    ProfileSyncRequest,
    ProfileSyncResult,
    RecognitionRequest,
    RecognitionResult,
    ShadowRecognitionScores,
    TrainingRequest,
    TrainingResult,
    config,
)
from speaker_recognition.recognizer import recognizer
from speaker_recognition.shadow import shadow_service
from speaker_recognition.warmup import WarmupStatus, warm_encoder

_LOGGER = logging.getLogger(__name__)
_RECOGNIZER_LOCK = Lock()
_SHADOW_LOCK = Lock()
_WARMUP_STATUS: WarmupStatus = warm_encoder(recognizer)


def _configured_trusted_local_hosts() -> set[str]:
    """Return normalized local host addresses supplied by the runtime."""
    configured = os.environ.get("TRUSTED_LOCAL_HOSTS", "172.30.32.1")
    trusted: set[str] = set()
    for value in configured.split(","):
        candidate = value.strip()
        if not candidate:
            continue
        try:
            trusted.add(str(ip_address(candidate)))
        except ValueError:
            continue
    return trusted or {"172.30.32.1"}


_TRUSTED_LOCAL_HOSTS = _configured_trusted_local_hosts()


class RequestBodyTooLarge(Exception):
    """Raised when an HTTP request exceeds the configured transport budget."""


class RequestBodyLimitMiddleware:
    """Reject oversized Content-Length and chunked HTTP request bodies."""

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_size:
                    response = JSONResponse(
                        {"detail": "Request body exceeds the 96 MiB limit"},
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_size:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            response = JSONResponse(
                {"detail": "Request body exceeds the 96 MiB limit"},
                status_code=413,
            )
            await response(scope, receive, send)


def _is_trusted_local(host: Optional[str]) -> bool:
    """Return whether a request source is local to the service/HA host."""
    if not host:
        return False
    try:
        address = ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or str(address) in _TRUSTED_LOCAL_HOSTS


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
    version="3.0.1",
    dependencies=[Depends(require_api_access)],
)
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_REQUEST_BODY_BYTES)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint, retrying only the authoritative encoder warm-up."""
    global _WARMUP_STATUS
    with _RECOGNIZER_LOCK:
        if not _WARMUP_STATUS.ready:
            _WARMUP_STATUS = warm_encoder(recognizer)
        trained = recognizer.is_trained
        enrolled_users = recognizer.enrolled_users
        engine_id = recognizer.engine_id
        engine_name = recognizer.engine_name
    # Never wait for the experimental lock in /health. A first ECAPA model
    # download or shadow training run may take much longer than the container
    # health-check timeout, but it must not make the authoritative service unhealthy.
    if shadow_service.enabled:
        shadow = shadow_service.recognizer
        shadow_engine_id = shadow.engine_id
        shadow_engine_name = shadow.engine_name
        shadow_trained = shadow.is_trained
        try:
            shadow_enrolled_users = shadow.enrolled_users
        except RuntimeError:
            shadow_enrolled_users = []
    else:
        shadow_engine_id = None
        shadow_engine_name = None
        shadow_trained = False
        shadow_enrolled_users = []
    shadow_error = shadow_service.last_error

    return HealthResponse(
        status="healthy" if _WARMUP_STATUS.ready else "degraded",
        trained=trained,
        enrolled_users=enrolled_users,
        encoder_ready=_WARMUP_STATUS.ready,
        warmup_seconds=_WARMUP_STATUS.seconds,
        warmup_error=_WARMUP_STATUS.error,
        engine_id=engine_id,
        engine_name=engine_name,
        shadow_engine_id=shadow_engine_id,
        shadow_engine_name=shadow_engine_name,
        shadow_trained=shadow_trained,
        shadow_enrolled_users=shadow_enrolled_users,
        shadow_error=shadow_error,
    )


@app.post(
    "/train",
    response_model=TrainingResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Training"],
)
def train(request: TrainingRequest) -> TrainingResult:
    """Train the authoritative speaker recognition model."""
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
    "/shadow/train",
    response_model=TrainingResult,
    responses={409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Experimental"],
)
def train_shadow(request: TrainingRequest) -> TrainingResult:
    """Train the optional shadow engine without touching authoritative profiles."""
    if not shadow_service.enabled:
        raise HTTPException(status_code=409, detail="No shadow engine is configured")
    try:
        with _SHADOW_LOCK:
            result = shadow_service.recognizer.train(request)
            shadow_service.clear_error()
            return result
    except ValueError as error:
        _LOGGER.warning("Shadow training rejected samples: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        with _SHADOW_LOCK:
            shadow_service.record_error(error)
        _LOGGER.exception("Experimental shadow training failed")
        raise HTTPException(status_code=500, detail="Shadow training failed") from error


@app.post(
    "/profiles/sync",
    response_model=ProfileSyncResult,
    responses={500: {"model": ErrorResponse}},
    tags=["Training"],
)
def sync_profiles(request: ProfileSyncRequest) -> ProfileSyncResult:
    """Remove authoritative profiles no longer present in HA configuration."""
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
    "/shadow/profiles/sync",
    response_model=ProfileSyncResult,
    responses={409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Experimental"],
)
def sync_shadow_profiles(request: ProfileSyncRequest) -> ProfileSyncResult:
    """Synchronize shadow profiles independently of authoritative profiles."""
    if not shadow_service.enabled:
        raise HTTPException(status_code=409, detail="No shadow engine is configured")
    try:
        with _SHADOW_LOCK:
            shadow = shadow_service.recognizer
            removed_users = shadow.sync_profiles(set(request.desired_users))
            shadow_service.clear_error()
            return ProfileSyncResult(
                enrolled_users=shadow.enrolled_users,
                removed_users=removed_users,
            )
    except Exception as error:
        with _SHADOW_LOCK:
            shadow_service.record_error(error)
        _LOGGER.exception("Error synchronizing experimental shadow profiles")
        raise HTTPException(status_code=500, detail="Shadow profile sync failed") from error


@app.post(
    "/recognize",
    response_model=RecognitionResult,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Recognition"],
)
def recognize(request: RecognitionRequest) -> RecognitionResult:
    """Recognize speaker using only the authoritative engine."""
    started = perf_counter()
    try:
        with _RECOGNIZER_LOCK:
            result = recognizer.recognize(request)
        result.processing_seconds = perf_counter() - started
        return result
    except (ValueError, RuntimeError) as error:
        _LOGGER.error("Recognition error: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        _LOGGER.exception("Error during recognition")
        raise HTTPException(
            status_code=500,
            detail="Speaker recognition failed",
        ) from error


@app.post(
    "/shadow/recognize",
    response_model=ShadowRecognitionScores,
    responses={409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Experimental"],
)
def recognize_shadow(request: RecognitionRequest) -> ShadowRecognitionScores:
    """Score audio with the shadow engine without applying an identity decision."""
    if not shadow_service.enabled:
        raise HTTPException(status_code=409, detail="No shadow engine is configured")
    started = perf_counter()
    try:
        with _SHADOW_LOCK:
            shadow = shadow_service.recognizer
            if not shadow.is_trained:
                raise HTTPException(status_code=409, detail="Shadow engine is not trained")
            scores = shadow.score(request)
            shadow_service.clear_error()
    except HTTPException:
        raise
    except (ValueError, RuntimeError) as error:
        _LOGGER.warning("Experimental shadow scoring failed: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        with _SHADOW_LOCK:
            shadow_service.record_error(error)
        _LOGGER.exception("Experimental shadow scoring failed")
        raise HTTPException(status_code=500, detail="Shadow scoring failed") from error
    return ShadowRecognitionScores(
        engine_id=scores.engine_id,
        candidate_user_id=scores.candidate_user_id,
        similarity=scores.similarity,
        margin=scores.margin,
        all_scores=scores.all_scores,
        processing_seconds=perf_counter() - started,
    )
