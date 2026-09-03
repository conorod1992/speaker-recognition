"""Runtime speaker recognition health validation."""

from __future__ import annotations

from typing import Any

from .live_evaluation import (
    LiveEvaluationSpeakerRecognition as _LiveEvaluationSpeakerRecognition,
)


class LiveEvaluationSpeakerRecognition(_LiveEvaluationSpeakerRecognition):
    """Validate authoritative encoder readiness on backend health reads."""

    async def _async_get(self, path: str) -> dict[str, Any]:
        """Reject health responses that cannot perform authoritative inference."""
        response = await super()._async_get(path)
        if path != "/health":
            return response

        encoder_ready = response.get("encoder_ready")
        if not isinstance(encoder_ready, bool):
            raise ValueError(
                "Speaker Recognition health response omitted encoder readiness"
            )
        if not encoder_ready:
            warmup_error = response.get("warmup_error")
            detail = (
                warmup_error
                if isinstance(warmup_error, str) and warmup_error
                else "encoder warm-up did not complete"
            )
            raise ValueError(f"Speaker Recognition encoder is not ready: {detail}")
        return response
