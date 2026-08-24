"""Concurrency regressions for STT audio fan-out."""

import asyncio
import importlib.util
from pathlib import Path

import pytest


def _load_stream_module():
    module_path = (
        Path(__file__).parents[1]
        / "custom_components"
        / "speaker_recognition"
        / "stream.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speaker_recognition_integration_stream", module_path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_recognition_overlaps_wrapped_stt_completion() -> None:
    """Recognition starts at incoming EOF while STT is still finalizing."""
    stream_module = _load_stream_module()
    recognition_started = asyncio.Event()
    allow_stt_to_finish = asyncio.Event()

    async def incoming_stream():
        yield b"first"
        yield b"second"

    async def stt_handler(stream):
        received = [chunk async for chunk in stream]
        assert received == [b"first", b"second"]
        await allow_stt_to_finish.wait()
        return "original transcription"

    async def recognition_handler(audio: bytes):
        assert audio == b"firstsecond"
        recognition_started.set()
        return "speaker"

    processing = asyncio.create_task(
        stream_module.async_process_buffered_stream(
            incoming_stream(), stt_handler, recognition_handler
        )
    )

    await asyncio.wait_for(recognition_started.wait(), timeout=1)
    assert not processing.done()
    allow_stt_to_finish.set()
    assert await processing == ("original transcription", "speaker")


@pytest.mark.asyncio
async def test_recognition_failure_preserves_stt_result() -> None:
    """Unexpected speaker failures cannot break a completed transcription."""
    stream_module = _load_stream_module()

    async def incoming_stream():
        yield b"audio"

    async def stt_handler(stream):
        async for _chunk in stream:
            pass
        return "transcription"

    async def recognition_handler(audio: bytes):
        del audio
        raise RuntimeError("backend unavailable")

    result = await stream_module.async_process_buffered_stream(
        incoming_stream(), stt_handler, recognition_handler
    )

    assert result == ("transcription", None)
