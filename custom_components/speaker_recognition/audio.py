"""Small, dependency-free audio conversions for the integration."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import wave

UNSUPPORTED_WAV_MESSAGE = (
    "Only uncompressed 16-bit PCM WAV files are currently supported."
)
MAX_UPLOADED_WAV_SECONDS = 30
MAX_LOCAL_WAV_BYTES = 16 * 1024 * 1024


def read_bounded_wav(path: Path) -> bytes:
    """Read a local WAV only after enforcing a stat-first size budget."""
    if path.stat().st_size > MAX_LOCAL_WAV_BYTES:
        raise ValueError("WAV file exceeds the 16 MiB upload limit")
    with path.open("rb") as stream:
        data = stream.read(MAX_LOCAL_WAV_BYTES + 1)
    if len(data) > MAX_LOCAL_WAV_BYTES:
        raise ValueError("WAV file exceeds the 16 MiB upload limit")
    return data


def decode_wav(
    audio_data: bytes, *, max_duration_seconds: int | None = MAX_UPLOADED_WAV_SECONDS
) -> tuple[bytes, int]:
    """Decode a PCM WAV file to signed 16-bit mono PCM."""
    try:
        with wave.open(BytesIO(audio_data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(UNSUPPORTED_WAV_MESSAGE)
            if wav_file.getsampwidth() != 2:
                raise ValueError(UNSUPPORTED_WAV_MESSAGE)

            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            frames = wav_file.getnframes()
            if (
                max_duration_seconds is not None
                and sample_rate > 0
                and frames > sample_rate * max_duration_seconds
            ):
                raise ValueError(
                    f"WAV audio must be {max_duration_seconds} seconds or shorter."
                )
            pcm_data = wav_file.readframes(frames)
        return pcm_to_mono(pcm_data, channels), sample_rate
    except (EOFError, wave.Error) as error:
        raise ValueError(UNSUPPORTED_WAV_MESSAGE) from error


def decode_persisted_training_wav(audio_data: bytes) -> tuple[bytes, int]:
    """Decode persisted enrollment media while tolerating legacy long recordings.

    New enrollment uploads remain subject to ``MAX_UPLOADED_WAV_SECONDS`` via
    ``decode_wav``. Existing configured media may predate that validation, so
    rebuilds decode the bounded local file and cap only the in-memory PCM sent to
    the backend. The original WAV on disk is never modified.
    """
    pcm_data, sample_rate = decode_wav(audio_data, max_duration_seconds=None)
    if sample_rate <= 0:
        return pcm_data, sample_rate
    max_pcm_bytes = sample_rate * 2 * MAX_UPLOADED_WAV_SECONDS
    return pcm_data[:max_pcm_bytes], sample_rate


def prepare_live_pcm(
    audio_data: bytes, sample_rate: int, channels: int
) -> tuple[bytes, int]:
    """Return mono PCM from a live STT buffer.

    Home Assistant STT streams with WAV metadata may contain either a complete
    WAV container or raw PCM frames. Decode only real WAV containers. Live Assist
    audio is not subject to the enrollment/upload duration cap.
    """
    if audio_data.startswith(b"RIFF") and audio_data[8:12] == b"WAVE":
        return decode_wav(audio_data, max_duration_seconds=None)
    return pcm_to_mono(audio_data, channels), sample_rate


def pcm_to_mono(pcm_data: bytes, channels: int) -> bytes:
    """Downmix little-endian signed 16-bit PCM to mono."""
    if channels < 1:
        raise ValueError("Audio stream has no channels")
    if len(pcm_data) % 2:
        raise ValueError("PCM audio has an incomplete 16-bit sample")
    if channels == 1:
        return pcm_data
    if len(pcm_data) % (channels * 2):
        raise ValueError("PCM audio has an incomplete audio frame")

    mono = bytearray()
    for offset in range(0, len(pcm_data), channels * 2):
        samples = [
            int.from_bytes(pcm_data[index : index + 2], "little", signed=True)
            for index in range(offset, offset + channels * 2, 2)
        ]
        mono.extend((sum(samples) // channels).to_bytes(2, "little", signed=True))
    return bytes(mono)
