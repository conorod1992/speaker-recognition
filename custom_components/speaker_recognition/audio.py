"""Small, dependency-free audio conversions for the integration."""

from __future__ import annotations

from io import BytesIO
import wave

UNSUPPORTED_WAV_MESSAGE = (
    "Only uncompressed 16-bit PCM WAV files are currently supported."
)


def decode_wav(audio_data: bytes) -> tuple[bytes, int]:
    """Decode a PCM WAV file to signed 16-bit mono PCM."""
    try:
        with wave.open(BytesIO(audio_data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError(UNSUPPORTED_WAV_MESSAGE)
            if wav_file.getsampwidth() != 2:
                raise ValueError(UNSUPPORTED_WAV_MESSAGE)

            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            pcm_data = wav_file.readframes(wav_file.getnframes())
        return pcm_to_mono(pcm_data, channels), sample_rate
    except (EOFError, ValueError, wave.Error) as error:
        raise ValueError(UNSUPPORTED_WAV_MESSAGE) from error


def prepare_live_pcm(
    audio_data: bytes, sample_rate: int, channels: int
) -> tuple[bytes, int]:
    """Return mono PCM from a live STT buffer.

    Home Assistant STT streams with WAV metadata may contain either a complete
    WAV container or raw PCM frames. Decode only real WAV containers.
    """
    if audio_data.startswith(b"RIFF") and audio_data[8:12] == b"WAVE":
        return decode_wav(audio_data)
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
