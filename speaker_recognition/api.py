"""FastAPI application for speaker recognition service."""

from __future__ import annotations

import base64
import binascii
from ipaddress import ip_address
import logging
import os
import secrets
from threading import Lock
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
    TrainingRequest,
    TrainingResult,
    config,
)
from speaker_recognition.recognizer import recognizer
from speaker_recognition.warmup import WarmupStatus, warm_encoder

_LOGGER = logging.getLogger(__name__)
_RECOGNIZER_LOCK = Lock()
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
    version="2.8.1",
    dependencies=[Depends(require_api_access)],
)
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_REQUEST_BODY_BYTES)


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
