"""Regression contract for authoritative recognition lock contention."""

from pathlib import Path


ROOT = Path(__file__).parents[1]
API = ROOT / "speaker_recognition" / "api.py"


def test_live_recognition_fails_fast_when_authoritative_lock_is_busy() -> None:
    """Live turns must not queue behind training or profile synchronization."""
    source = API.read_text(encoding="utf-8")
    recognize_body = source.split("def recognize(", 1)[1].split("\n\n@app.post(", 1)[0]

    assert "_RECOGNIZER_LOCK.acquire(blocking=False)" in recognize_body
    assert "status_code=503" in recognize_body
    assert "_RECOGNIZER_LOCK.release()" in recognize_body


def test_recognition_endpoint_documents_busy_response() -> None:
    """The API schema should expose transient model contention as unavailable."""
    source = API.read_text(encoding="utf-8")
    decorator = source.split('    "/recognize",', 1)[1].split("def recognize(", 1)[0]

    assert '503: {"model": ErrorResponse}' in decorator
