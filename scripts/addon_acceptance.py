#!/usr/bin/env python3
"""Acceptance probe for the real Speaker Recognition add-on container."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

SAMPLE_RATE = 16000
USER_ID = "acceptance-alice"


def _request(
    base_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
    expected_status: int = 200,
) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}", data=data, headers=headers
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        status = error.code
        body = error.read().decode("utf-8")
    if status != expected_status:
        raise AssertionError(
            f"{path}: expected HTTP {expected_status}, got {status}: {body}"
        )
    return json.loads(body) if body else {}


def _audio_payload(path: Path) -> dict[str, object]:
    pcm = path.read_bytes()
    if not pcm:
        raise AssertionError(f"empty PCM fixture: {path}")
    return {
        "audio_data": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": SAMPLE_RATE,
    }


def wait_healthy(base_url: str, token: str) -> dict:
    deadline = time.monotonic() + 240
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = _request(base_url, "/health", token=token)
            if health.get("status") == "healthy" and health.get("encoder_ready") is True:
                return health
        except Exception as error:  # noqa: BLE001 - surfaced after bounded retry
            last_error = error
        time.sleep(2)
    raise AssertionError(f"add-on never became healthy: {last_error}")


def initial_phase(base_url: str, token: str, audio_dir: Path) -> None:
    health = wait_healthy(base_url, token)
    assert health["trained"] is False

    # Requests entering through Docker's published port are not loopback inside the
    # container, so a configured token must protect the API.
    unauthorized = _request(base_url, "/health", expected_status=401)
    assert "authentication required" in unauthorized.get("detail", "").lower()

    # Exercise the exact profile-sync shape used by the Home Assistant integration.
    sync = _request(
        base_url,
        "/profiles/sync",
        token=token,
        payload={"desired_users": []},
    )
    assert sync == {"enrolled_users": [], "removed_users": []}

    training_paths = [audio_dir / f"train-{index}.pcm" for index in range(1, 4)]
    train = _request(
        base_url,
        "/train",
        token=token,
        payload={
            "voice_samples": [
                {"user": USER_ID, "audio": _audio_payload(path)}
                for path in training_paths
            ]
        },
    )
    assert USER_ID in train["trained_users"]
    assert train["accepted_samples"][USER_ID] >= 3

    recognized = _request(
        base_url,
        "/recognize",
        token=token,
        payload={"audio": _audio_payload(audio_dir / "recognize.pcm")},
    )
    assert recognized["candidate_user_id"] == USER_ID
    assert recognized["accepted"] is True
    assert recognized["user_id"] == USER_ID

    health = _request(base_url, "/health", token=token)
    assert health["trained"] is True
    assert USER_ID in health["enrolled_users"]


def restart_phase(base_url: str, token: str, audio_dir: Path) -> None:
    health = wait_healthy(base_url, token)
    assert health["trained"] is True
    assert USER_ID in health["enrolled_users"]

    recognized = _request(
        base_url,
        "/recognize",
        token=token,
        payload={"audio": _audio_payload(audio_dir / "recognize.pcm")},
    )
    assert recognized["accepted"] is True
    assert recognized["user_id"] == USER_ID

    removed = _request(
        base_url,
        "/profiles/sync",
        token=token,
        payload={"desired_users": []},
    )
    assert USER_ID in removed["removed_users"]
    assert USER_ID not in removed["enrolled_users"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("initial", "restart"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18099")
    parser.add_argument("--token", default="addon-acceptance-token")
    parser.add_argument("--audio-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "initial":
        initial_phase(args.base_url, args.token, args.audio_dir)
    else:
        restart_phase(args.base_url, args.token, args.audio_dir)


if __name__ == "__main__":
    main()
