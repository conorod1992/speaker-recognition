"""Record the Codex findings addressed by the transactional enrollment fix."""

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_transactional_enrollment_fix_covers_findings_one_and_five() -> None:
    enrollment = (ROOT / "custom_components/speaker_recognition/enrollment.py").read_text(
        encoding="utf-8"
    )
    websocket = (ROOT / "custom_components/speaker_recognition/websocket.py").read_text(
        encoding="utf-8"
    )
    lifecycle = (ROOT / "custom_components/speaker_recognition/lifecycle.py").read_text(
        encoding="utf-8"
    )

    assert "sample_{sample_index + 1}_{generation}.wav" in enrollment
    assert '"enrollment_commit_waiters"' in websocket
    assert "async_apply_enrollment_update" in lifecycle
