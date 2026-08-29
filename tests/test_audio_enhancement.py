"""Tests for the experimental dependency-free speech enhancement preview."""

from array import array
import math

from custom_components.speaker_recognition.enhancement import enhance_speech_pcm


def _tone(frequency: float, sample_rate: int = 16000, seconds: float = 1.0) -> bytes:
    samples = array(
        "h",
        (
            int(8000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            for index in range(int(sample_rate * seconds))
        ),
    )
    return samples.tobytes()


def _rms(pcm_data: bytes) -> float:
    samples = array("h")
    samples.frombytes(pcm_data)
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def test_enhancement_preserves_pcm_shape_and_speech_band_energy() -> None:
    """Enhancement keeps PCM framing intact and retains ordinary speech-band audio."""
    original = _tone(1000.0)
    enhanced = enhance_speech_pcm(original, 16000)

    assert len(enhanced) == len(original)
    assert _rms(enhanced) > 1000


def test_enhancement_reduces_50_hz_mains_component_relative_to_speech() -> None:
    """The mains notch should suppress 50 Hz more strongly than 1 kHz speech."""
    mains = _tone(50.0)
    speech = _tone(1000.0)

    mains_ratio = _rms(enhance_speech_pcm(mains, 16000)) / _rms(mains)
    speech_ratio = _rms(enhance_speech_pcm(speech, 16000)) / _rms(speech)

    assert mains_ratio < speech_ratio


def test_invalid_pcm_is_returned_unchanged() -> None:
    """Malformed or unsupported PCM input fails safe."""
    assert enhance_speech_pcm(b"odd", 16000) == b"odd"
    assert enhance_speech_pcm(b"\x00\x00", 0) == b"\x00\x00"
