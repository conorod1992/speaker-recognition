#!/usr/bin/env python3
"""Acceptance probe for the real Speaker Recognition add-on container."""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import struct
import time
import urllib.error
import urllib.request

SAMPLE_RATE = 16000
USER_ID = "acceptance-alice"


def _pcm(seed: int, *, seconds: float = 3.6) -> bytes:
    """Generate deterministic speech-like PCM16 with voiced/silent syllable structure."""
    rng = random.Random(seed)
    total = int(SAMPLE_RATE * seconds)
    samples: list[int] = []
    for index in range(total):
        t = index / SAMPLE_RATE
        syllable = int(t / 0.24)
        phase = t % 0.24
        # Short inter-syllable gaps help Resemblyzer/WebRTC VAD see a speech-like cadence.
        if phase > 0.205:
            samples.append(0)
            continue
        f0 = 108.0 + (syllable % 5) * 6.0 + (seed % 3) * 1.5
        envelope = min(1.0, phase / 0.025, (0.205 - phase) / 0.03)
        value = 0.0
        # Harmonic stack with broad formant-like emphasis rather than a single test tone.
        for harmonic in range(1, 19):
            frequency = f0 * harmonic
            formant = (
                math.exp(-((frequency - 650.0) / 330.0) ** 2)
                + 0.7 * math.exp(-((frequency - 1250.0) / 450.0) ** 2)
                + 0.35 * math.exp(-((frequency - 2450.0) / 700.0) ** 2)
                + 0.08
            )
            value += (formant / harmonic) * math.sin(
                2.0 * math.pi * frequency * t + 0.07 * seed * harmonic
            )
        value += rng.uniform(-0.025, 0.025)
        sample = int(max(-1.0, min(1.0, value * 0.19 * envelope)) * 32767)
        samples.append(sample)
    return b"".join(struct.pack("<h", sample) for sample in samples)


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
        with urllib.request.urlopen(request, timeout=30) as response:
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


def _audio_payload(seed: int) -> dict[str, object]:
    return {
        "audio_data": base64.b64encode(_pcm(seed)).decode("ascii"),
        "sample_rate": SAMPLE_RATE,
    }


def wait_healthy(base_url: str, token: str) -> dict:
    deadline = time.monotonic() + 210
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


def initial_phase(base_url: str, token: str) -> None:
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

    train = _request(
        base_url,
        "/train",
        token=token,
        payload={
            "voice_samples": [
                {"user": USER_ID, "audio": _audio_payload(seed)}
                for seed in (11, 12, 13)
            ]
        },
    )
    assert USER_ID in train["trained_users"]
    assert train["accepted_samples"][USER_ID] >= 3

    recognized = _request(
        base_url,
        "/recognize",
        token=token,
        payload={"audio": _audio_payload(11)},
    )
    assert recognized["candidate_user_id"] == USER_ID
    assert recognized["accepted"] is True
    assert recognized["user_id"] == USER_ID

    health = _request(base_url, "/health", token=token)
    assert health["trained"] is True
    assert USER_ID in health["enrolled_users"]


def restart_phase(base_url: str, token: str) -> None:
    health = wait_healthy(base_url, token)
    assert health["trained"] is True
    assert USER_ID in health["enrolled_users"]

    recognized = _request(
        base_url,
        "/recognize",
        token=token,
        payload={"audio": _audio_payload(11)},
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
    args = parser.parse_args()
    if args.phase == "initial":
        initial_phase(args.base_url, args.token)
    else:
        restart_phase(args.base_url, args.token)


if __name__ == "__main__":
    main()
