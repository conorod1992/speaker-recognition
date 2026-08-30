"""Race-safe audio stream fan-out for STT and speaker recognition."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Awaitable, Callable
import logging
from typing import TypeVar

_LOGGER = logging.getLogger(__name__)

SttResultT = TypeVar("SttResultT")
RecognitionResultT = TypeVar("RecognitionResultT")

MAX_RECOGNITION_BUFFER_BYTES = 32 * 1024 * 1024


async def async_process_buffered_stream(
    stream: AsyncIterable[bytes],
    stt_handler: Callable[[AsyncIterable[bytes]], Awaitable[SttResultT]],
    recognition_handler: Callable[[bytes], Awaitable[RecognitionResultT]],
) -> tuple[SttResultT, RecognitionResultT | None]:
    """Feed STT immediately and start recognition when input reaches EOF.

    The wrapped STT controls backpressure while yielded chunks are copied into a
    bounded utterance buffer. If the recognition buffer limit is exceeded, STT
    continues normally but supplemental speaker recognition is skipped for that
    utterance instead of allowing unbounded memory growth.
    """
    audio_buffer = bytearray()
    buffer_overflowed = False
    stream_finished = asyncio.Event()
    stream_was_fully_consumed = False

    async def buffered_stream() -> AsyncIterable[bytes]:
        nonlocal buffer_overflowed, stream_was_fully_consumed
        async for chunk in stream:
            if not buffer_overflowed:
                remaining = MAX_RECOGNITION_BUFFER_BYTES - len(audio_buffer)
                if len(chunk) <= remaining:
                    audio_buffer.extend(chunk)
                else:
                    buffer_overflowed = True
                    audio_buffer.clear()
                    _LOGGER.warning(
                        "Skipping speaker recognition because buffered STT audio "
                        "exceeded %d bytes",
                        MAX_RECOGNITION_BUFFER_BYTES,
                    )
            yield chunk
        stream_was_fully_consumed = True
        stream_finished.set()

    async def recognize_after_eof() -> RecognitionResultT | None:
        await stream_finished.wait()
        if (
            not stream_was_fully_consumed
            or buffer_overflowed
            or not audio_buffer
        ):
            return None
        try:
            return await recognition_handler(bytes(audio_buffer))
        except Exception:
            # Speaker recognition is supplemental and must never invalidate STT.
            _LOGGER.exception("Speaker recognition failed after audio buffering")
            return None

    recognition_task = asyncio.create_task(
        recognize_after_eof(), name="speaker-recognition-after-audio-eof"
    )
    try:
        stt_result = await stt_handler(buffered_stream())
    except BaseException:
        stream_finished.set()
        recognition_task.cancel()
        await asyncio.gather(recognition_task, return_exceptions=True)
        raise
    else:
        # A provider may return without exhausting its input. Wake the
        # recognition task so it can safely skip an incomplete utterance.
        stream_finished.set()

    return stt_result, await recognition_task
