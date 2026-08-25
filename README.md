# Speaker Recognition for Home Assistant

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-integration-blue)

**Identify who is speaking to Home Assistant using their voice.**

Speaker Recognition can learn the voices of Home Assistant users and identify
which enrolled user is speaking during voice interactions.

It can be used alongside your existing speech-to-text (STT) provider and
conversation agent, allowing other integrations or automations to make use of
the detected speaker.

> [!IMPORTANT]
> For a normal Home Assistant installation, Speaker Recognition has **two
> parts**:
>
> 1. The **Speaker Recognition app/add-on**, which performs the actual voice
>    matching.
> 2. The **Speaker Recognition integration**, installed through HACS, which
>    connects Home Assistant to the app and provides enrollment, STT and
>    conversation features.
>
> Most Home Assistant users should install **both**.

## Features

- 🎤 Identifies enrolled users from their voice
- 🏠 Integrates directly with Home Assistant
- 👤 Guided voice enrollment for Home Assistant users
- 🗣️ Works alongside an existing speech-to-text provider
- 💬 Works alongside an existing conversation agent
- 🔄 Users can be retrained without affecting other enrolled users
- 💾 Trained voice profiles are retained across restarts
- 🐳 Can also run separately using Docker
- 🔌 REST API and Python client available for advanced use

---

# Installation

## Home Assistant OS / Supervised

This is the recommended setup if your Home Assistant installation supports
apps/add-ons.

You will install the Speaker Recognition app first, then install the Home
Assistant integration through HACS.

### 1. Install the Speaker Recognition app

1. In Home Assistant, open **Settings → Apps**.

   On older Home Assistant versions this may be called **Add-ons** or
   **Add-on Store**.

2. Open the **three-dot menu** and select **Repositories**.

3. Add:

   ```text
   https://github.com/conorod1992/speaker-recognition
   ```

4. Find and install **Speaker Recognition**.

5. The default settings should be suitable for most installations:

   ```yaml
   host: "0.0.0.0"
   port: 8099
   log_level: "info"
   access_log: true
   embeddings_dir: "/share/speaker_recognition/embeddings"
   ```

6. Start the app.

The first installation may take several minutes because the recognition
software and its machine-learning dependencies need to be built.

The app currently supports:

- `amd64`
- `aarch64`

Your trained voice profiles are stored under
`/share/speaker_recognition/embeddings`, so they are retained when the app or
Home Assistant restarts.

### 2. Install the Home Assistant integration through HACS

If you have not used a custom HACS repository before:

1. Open **HACS**.
2. Open **Integrations**.
3. Select the **three-dot menu → Custom repositories**.
4. Enter:

   ```text
   https://github.com/conorod1992/speaker-recognition
   ```

5. Select **Integration** as the repository type.
6. Add the repository.
7. Find **Speaker Recognition** in HACS and select **Download**.
8. Restart Home Assistant.

You do **not** need to manually copy anything into
`custom_components`.

### 3. Add Speaker Recognition to Home Assistant

After restarting Home Assistant:

1. Open **Settings → Devices & services**.
2. Select **Add integration**.
3. Search for **Speaker Recognition**.
4. Enter the address of the Speaker Recognition app.

For example:

```text
http://192.168.1.100:8099
```

Replace `192.168.1.100` with the IP address of your Home Assistant system.

> [!TIP]
> If Home Assistant and the Speaker Recognition app are running on the same
> machine, this is still the address of the Speaker Recognition service — not
> a web page you need to use during normal operation.

Once Home Assistant successfully connects, you can enroll your first speaker.

---

# Enrolling a speaker

Speaker Recognition needs several short recordings of each person before it
can identify them.

Enrollment is handled through the Home Assistant interface.

1. Open the **Speaker Recognition** integration.
2. Choose **Enroll or retrain a user**.
3. Select the Home Assistant user whose voice you want to enroll.
4. Home Assistant will show a short phrase to read aloud.
5. Record the phrase naturally.
6. Upload or select the recording when prompted.
7. Repeat for the remaining phrases.
8. Finish enrollment.

Five recordings are sufficient, with a sixth suggested recording available.

Each recording must be a:

- WAV file
- uncompressed
- 16-bit PCM recording
- at least 0.5 seconds long

You can replace an individual recording before finishing enrollment if you
are unhappy with it.

### Recording your samples

Home Assistant's custom-integration setup screens do not currently provide a
general microphone-recording control.

You therefore need to record the phrase first and then select/upload the WAV
file when prompted.

You can use, for example:

- your phone
- a computer
- a recording application
- the Home Assistant companion app where suitable

For the best results, try to record the samples using a microphone and
environment reasonably similar to those you expect to use for voice commands.

Speak naturally rather than deliberately changing your voice for the
recordings.

### Retraining a user

You can repeat enrollment later if recognition for a particular person needs
improvement.

Retraining one Home Assistant user replaces only that user's voice profile.
Other enrolled users are left unchanged.

---

# Using Speaker Recognition with Home Assistant voice

Speaker Recognition does **not** replace your existing speech-to-text provider
or conversation agent.

Instead, it can sit alongside them and add speaker identification.

There are two optional helpers you can add.

## STT proxy

The **STT proxy** wraps an existing Home Assistant speech-to-text entity.

Your normal STT provider still converts speech into text, while Speaker
Recognition also analyses the same voice input to determine who was speaking.

To add it:

1. Open the Speaker Recognition integration.
2. Choose **Add STT proxy**.
3. Select the speech-to-text entity you already use.

Speaker recognition and speech-to-text processing are performed in parallel
where possible, reducing the additional delay.

If speaker recognition fails, the original STT result is still returned.

## Conversation proxy

The **Conversation proxy** wraps an existing Home Assistant conversation
agent.

It allows the conversation to include information about the detected speaker.

To add it:

1. Open the Speaker Recognition integration.
2. Choose **Add Conversation proxy**.
3. Select your existing conversation agent.
4. Set the **Minimum confidence** required before a detected speaker is used.

Your original conversation agent continues to handle the request.

---

# How speaker recognition works

You do not need to understand the recognition system to use the integration,
but in simplified terms:

1. You provide several recordings of a person's voice.
2. Speaker Recognition converts those recordings into a mathematical
   representation of that person's voice.
3. When someone speaks later, their voice is compared with the enrolled
   profiles.
4. The closest match is returned along with a confidence score.

These stored voice representations are sometimes called **voice embeddings**
or **voiceprints**.

The recognition engine uses
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer).

---

# Troubleshooting

## Speaker Recognition does not appear in HACS

Make sure the repository was added as a **Custom repository** with the type
set to **Integration**:

```text
https://github.com/conorod1992/speaker-recognition
```

Then search for **Speaker Recognition** again in HACS.

## Speaker Recognition does not appear under Devices & services

After downloading a custom integration through HACS, Home Assistant normally
needs to be restarted before the integration becomes available.

Restart Home Assistant and then go to:

**Settings → Devices & services → Add integration**

## Home Assistant cannot connect to the backend

Check that:

- the Speaker Recognition app is running;
- port `8099` is available;
- the backend URL contains the correct Home Assistant/server IP address;
- the URL begins with `http://` unless you have separately configured HTTPS.

You can also open:

```text
http://HOME_ASSISTANT_IP:8099/health
```

A healthy service should return JSON containing:

```json
{
  "status": "healthy"
}
```

## A voice sample is rejected

Enrollment recordings must be local, uncompressed **16-bit PCM WAV** files
containing at least 0.5 seconds of audio.

Converting an MP3 to WAV by simply changing the file extension is not enough;
the audio itself must be encoded as PCM WAV.

## Recognition is unreliable

Speaker recognition is affected by recording conditions.

For better results:

- use clear recordings with little background noise;
- speak naturally during enrollment;
- avoid recordings that are extremely quiet or distorted;
- use enrollment recordings made in conditions reasonably similar to normal
  voice use;
- retrain the affected user if their original samples were poor.

Recognition confidence should not be treated as absolute proof of someone's
identity.

---

# Updating

## Updating the integration

Updates to the Home Assistant integration are delivered through HACS.

Open **HACS** and install an available Speaker Recognition update in the same
way as other custom integrations.

Restart Home Assistant if prompted.

## Updating the app

Updates to the Speaker Recognition app appear through Home Assistant's app /
add-on update system.

Your enrolled voice profiles are stored separately from the application and
should remain available after ordinary updates or restarts.

---

# Advanced installation

The sections below are intended for users running Speaker Recognition outside
Home Assistant OS/Supervised, or for developers integrating directly with the
service.

## Docker

The recognition backend can be run as a standalone Docker container:

```bash
docker run -d \
  -p 8099:8099 \
  -v ./embeddings:/app/embeddings \
  ghcr.io/conorod1992/speaker-recognition:latest
```

Then configure the Home Assistant integration to use the address of that
server, for example:

```text
http://192.168.1.50:8099
```

---

# REST API

The recognition backend also exposes a REST API for advanced integrations.

## `GET /health`

Returns the current backend status and information about persisted enrolled
users.

Example:

```json
{
  "status": "healthy",
  "trained": true,
  "enrolled_users": [
    "Alice",
    "Bob"
  ]
}
```

Home Assistant uses this persisted status during setup. A normal Home Assistant
restart or integration reload does not retrain every enrolled user.

## `POST /train`

Creates or replaces voice profiles from supplied audio samples.

Example request:

```json
{
  "voice_samples": [
    {
      "user": "Alice",
      "audio_input": {
        "audio_data": "base64-string",
        "sample_rate": 16000
      }
    }
  ]
}
```

Example response:

```json
{
  "status": "success",
  "trained_users": [
    "Alice"
  ],
  "count": 2,
  "accepted_samples": {
    "Alice": 6
  },
  "rejected_samples": {
    "Alice": 0
  }
}
```

Training an existing user replaces that user's recognition profile while
preserving profiles belonging to other users.

## `POST /recognize`

Attempts to identify the speaker in supplied audio.

Example request:

```json
{
  "audio_input": {
    "audio_data": "base64-string",
    "sample_rate": 16000
  }
}
```

Example response:

```json
{
  "speaker": "Alice",
  "confidence": 0.95
}
```

---

# Python client

The project also contains a Python client for applications that need to
communicate with the Speaker Recognition service directly.

Install the client package with:

```bash
pip install hass-speaker-recognition
```

To include the recognition server dependencies:

```bash
pip install "hass-speaker-recognition[server]"
```

The server dependencies currently require Python below 3.10.

Example recognition request:

```python
from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import AudioInput, RecognitionRequest

async with SpeakerRecognitionClient("http://localhost:8099") as client:
    result = await client.recognize(
        RecognitionRequest(
            audio_input=AudioInput(
                audio_data="<base64-encoded-audio>",
                sample_rate=16000,
            )
        )
    )

    print(
        f"Speaker: {result.speaker} "
        f"(confidence: {result.confidence:.2%})"
    )
```

---

# Backend configuration

The Home Assistant app uses these default settings:

```yaml
host: "0.0.0.0"
port: 8099
log_level: "info"
access_log: true
embeddings_dir: "/share/speaker_recognition/embeddings"
```

For standalone installations, the backend also supports the following
environment variables:

| Variable | Purpose | Default |
| --- | --- | --- |
| `HOST` | Address the server listens on | `0.0.0.0` |
| `PORT` | API port | `8099` |
| `LOG_LEVEL` | Logging level | `info` |
| `ACCESS_LOG` | Enable HTTP access logging | `true` |
| `EMBEDDINGS_DIR` | Voice-profile storage directory | `./embeddings` |

---

# Development

## Requirements

For server development:

- Python 3.9
- [uv](https://github.com/astral-sh/uv)

The client-only package supports Python 3.8 and later.

## Set up the development environment

```bash
git clone https://github.com/conorod1992/speaker-recognition.git
cd speaker-recognition

uv sync --all-groups
```

Run the tests:

```bash
uv run pytest tests/ -v
```

Run linting:

```bash
uv run ruff check .
```

Run type checking:

```bash
uv run mypy --strict speaker_recognition
```

## Running the server locally

```bash
uv run python -m speaker_recognition
```

Or specify the host and port:

```bash
uv run python -m speaker_recognition \
  --host 0.0.0.0 \
  --port 8099
```

## Add-on source synchronization

The Home Assistant app/add-on contains a committed copy of the recognition
package used as its Supervisor Docker build context.

After changing files under `speaker_recognition/`, synchronize that copy with:

```bash
python scripts/sync_addon_sources.py
```

Verify that it has not drifted with:

```bash
python scripts/sync_addon_sources.py --check
```

---

# Project structure

```text
speaker-recognition/
├── speaker_recognition/          # Recognition backend and Python client
├── custom_components/
│   └── speaker_recognition/      # Home Assistant integration
├── speaker_recognition_addon/    # Home Assistant app/add-on
├── tests/                        # Test suite
└── example_data/                 # Example audio files
```

---

# Contributing

Contributions, bug reports and feature suggestions are welcome.

If contributing code:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Run the relevant tests and checks.
5. Commit your changes.
6. Push your branch.
7. Open a pull request.

Please:

- use descriptive names;
- add type annotations where appropriate;
- add or update tests for behavioural changes;
- keep changes focused where possible.

---

# Support

If something is not working, or you have an idea for the project:

- [Report a bug](https://github.com/conorod1992/speaker-recognition/issues)
- [Request a feature](https://github.com/conorod1992/speaker-recognition/issues)

When reporting a problem, including relevant Home Assistant and Speaker
Recognition logs can make diagnosis much easier.

---

# License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

Speaker Recognition makes use of:

- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker voice
  embeddings
- [Home Assistant](https://www.home-assistant.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

Made with ❤️ for the Home Assistant community
