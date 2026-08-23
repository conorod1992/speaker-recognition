"""Small, dependency-free audio conversions for the integration."""

from __future__ import annotations

from io import BytesIO
import wave


def decode_wav(audio_data: bytes) -> tuple[bytes, int]:
    """Decode a PCM WAV file to signed 16-bit mono PCM."""
    try:
        with wave.open(BytesIO(audio_data), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("Only uncompressed PCM WAV files are supported")
            if wav_file.getsampwidth() != 2:
                raise ValueError("Only 16-bit PCM WAV audio is supported")

            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            pcm_data = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError("Invalid WAV audio") from error

    return pcm_to_mono(pcm_data, channels), sample_rate


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
