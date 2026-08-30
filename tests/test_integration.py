"""Integration tests for speaker recognition API and client."""

import base64
import socket
import wave
from multiprocessing import Process
from pathlib import Path

import pytest
import uvicorn

from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import (
    AudioInput,
    RecognitionRequest,
    TrainingRequest,
    VoiceSample,
)

EXAMPLE_DATA_DIR = Path(__file__).parent.parent / "example_data"
API_HOST = "127.0.0.1"


def start_api_server(port: int) -> None:
    """Start the API server in a subprocess."""
    from speaker_recognition.api import app

    uvicorn.run(app, host=API_HOST, port=port, log_level="error")


def read_audio_file_as_base64(file_path: Path) -> tuple[str, int]:
    """Read WAV file and encode PCM data as base64."""
    with wave.open(str(file_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        pcm_data = wav_file.readframes(wav_file.getnframes())
        return base64.b64encode(pcm_data).decode("utf-8"), sample_rate


@pytest.fixture(scope="module")
def api_server() -> str:
    """Start API server for testing."""
    import time

    import httpx

    with socket.socket() as server_socket:
        server_socket.bind((API_HOST, 0))
        port = server_socket.getsockname()[1]

    api_base_url = f"http://{API_HOST}:{port}"
    server_process = Process(target=start_api_server, args=(port,))
    server_process.start()

    for attempt in range(30):
        try:
            response = httpx.get(f"{api_base_url}/health", timeout=1.0)
            if response.status_code == 200:
                break
        except (httpx.ConnectError, httpx.TimeoutException):
            if attempt == 29:
                server_process.terminate()
                server_process.join(timeout=5)
                raise RuntimeError("Server failed to start within timeout")
            time.sleep(0.5)

    yield api_base_url
    server_process.terminate()
    server_process.join(timeout=5)
    if server_process.is_alive():
        server_process.kill()


@pytest.mark.asyncio
async def test_train_and_recognize_speakers(api_server: str):
    """Test robust multi-sample training and recognition for two speakers."""
    speaker1_training_file = EXAMPLE_DATA_DIR / "speaker1_1.wav"
    speaker2_training_file = EXAMPLE_DATA_DIR / "speaker2_1.wav"
    speaker1_recognition_file = EXAMPLE_DATA_DIR / "speaker1_2.wav"
    speaker2_recognition_file = EXAMPLE_DATA_DIR / "speaker2_2.wav"

    for path in (
        speaker1_training_file,
        speaker2_training_file,
        speaker1_recognition_file,
        speaker2_recognition_file,
    ):
        assert path.exists(), f"Missing {path}"

    async with SpeakerRecognitionClient(api_server, timeout=60.0) as client:
        health = await client.health_check()
        assert health.status == "healthy"

        speaker1_audio_data, speaker1_rate = read_audio_file_as_base64(
            speaker1_training_file
        )
        speaker2_audio_data, speaker2_rate = read_audio_file_as_base64(
            speaker2_training_file
        )
        speaker1_sample = VoiceSample(
            user="speaker1",
            audio=AudioInput(
                audio_data=speaker1_audio_data, sample_rate=speaker1_rate
            ),
        )
        speaker2_sample = VoiceSample(
            user="speaker2",
            audio=AudioInput(
                audio_data=speaker2_audio_data, sample_rate=speaker2_rate
            ),
        )
        training_request = TrainingRequest(
            voice_samples=[speaker1_sample] * 3 + [speaker2_sample] * 3
        )

        training_result = await client.train(training_request)
        assert training_result.status == "success"
        assert training_result.count >= 2
        assert {"speaker1", "speaker2"} <= set(training_result.trained_users)
        assert training_result.accepted_samples["speaker1"] >= 3
        assert training_result.accepted_samples["speaker2"] >= 3

        persisted_status = await client.health_check()
        assert persisted_status.trained
        assert {"speaker1", "speaker2"} <= set(persisted_status.enrolled_users)

        speaker1_recognition_audio, speaker1_rec_rate = read_audio_file_as_base64(
            speaker1_recognition_file
        )
        recognition_result_1 = await client.recognize(
            RecognitionRequest(
                audio=AudioInput(
                    audio_data=speaker1_recognition_audio,
                    sample_rate=speaker1_rec_rate,
                )
            )
        )
        assert recognition_result_1.user_id == "speaker1", (
            f"Expected speaker1, got {recognition_result_1.user_id} "
            f"with confidence {recognition_result_1.confidence}. "
            f"All scores: {recognition_result_1.all_scores}"
        )

        speaker2_recognition_audio, speaker2_rec_rate = read_audio_file_as_base64(
            speaker2_recognition_file
        )
        recognition_result_2 = await client.recognize(
            RecognitionRequest(
                audio=AudioInput(
                    audio_data=speaker2_recognition_audio,
                    sample_rate=speaker2_rec_rate,
                )
            )
        )
        assert recognition_result_2.user_id == "speaker2", (
            f"Expected speaker2, got {recognition_result_2.user_id} "
            f"with confidence {recognition_result_2.confidence}. "
            f"All scores: {recognition_result_2.all_scores}"
        )
