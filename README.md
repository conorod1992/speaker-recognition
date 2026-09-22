# Speaker Recognition for Home Assistant

![Python Version](https://img.shields.io/badge/python-3.9-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Home Assistant](https://img.shields.io/badge/Home%20Assistant-integration-blue)

**Identify who is speaking to Home Assistant using their voice.**

Speaker Recognition can learn the voices of Home Assistant users and identify
which enrolled user is speaking during voice interactions.

It works alongside your existing speech-to-text (STT) provider and conversation
agent. Speaker recognition runs in parallel with STT where possible, then the
conversation proxy can attach the recognised Home Assistant user to that exact
Assist turn.

> [!IMPORTANT]
> For a normal Home Assistant installation, Speaker Recognition has **two
> parts**:
>
> 1. The **Speaker Recognition app/add-on**, which performs the actual voice
>    matching.
> 2. The **Speaker Recognition integration**, installed through HACS, which
>    connects Home Assistant to the app and provides enrollment, STT,
>    conversation and diagnostics features.
>
> Most Home Assistant users should install **both**.

## Features

- 🎤 Identifies enrolled Home Assistant users from their voice
- 👤 Guided enrollment and retraining from a Home Assistant sidebar panel
- 🎙️ Records enrollment samples in the browser when microphone access is available
- 🛰️ Can capture enrollment samples through compatible Assist satellites
- 🗣️ Runs alongside an existing speech-to-text provider
- 💬 Runs alongside an existing conversation agent
- 🔗 Correlates recognition to the exact Assist turn
- ⚡ Runs recognition and STT concurrently where possible
- 🧪 Live satellite testing using the normal wake-word and Assist path
- 📊 Shows recognition timing, similarity, margin and added Assist latency
- 🧭 Stores a bounded, audio-free history of real recognition decisions
- ✅ Accepts explicit feedback on correct, wrong-speaker and missed decisions
- 🎚️ Can recommend a Home Assistant confidence threshold from labelled real use
- 🔄 Retraining one user does not replace other users' profiles
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

Release versions use prebuilt `amd64` and `aarch64` images, so Home Assistant
normally downloads the matching image rather than rebuilding the machine-learning
stack on the device.

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

You do **not** need to manually copy anything into `custom_components`.

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

Once Home Assistant successfully connects, a **Speaker Recognition** panel is
available in the Home Assistant sidebar.

---

# Enrolling a speaker

Speaker Recognition needs several short recordings of each person before it
can identify them.

For the easiest enrollment flow:

1. Open **Speaker Recognition** from the Home Assistant sidebar.
2. Under **Enroll or retrain a voice**, select the Home Assistant user.
3. Read the displayed phrase naturally.
4. Capture the sample with either:
   - **Record with this device**, when your browser allows microphone access; or
   - **Record with a voice satellite**, for a compatible Assist satellite.
5. Repeat for the remaining phrases.
6. When at least five samples are staged, select **Train**.

Five recordings are sufficient. A sixth phrase is available if you want an
extra sample.

### Browser microphone enrollment

Browser recording uses the normal browser microphone permission. Most browsers
only expose microphone APIs in a secure context, normally **HTTPS** or
`localhost`.

If microphone access is unavailable, the panel explains this and you can use a
compatible Assist satellite instead. The older local WAV selection flow in the
integration options remains available as another fallback.

### Satellite enrollment

Compatible Assist satellites can be prompted directly from the panel. The
integration binds the capture to the selected satellite and exact Assist turn,
so unrelated speech from another satellite is ignored.

For useful training data, speak naturally rather than deliberately changing
your voice. Recording conditions reasonably similar to normal Assist use are
usually preferable.

### Retraining a user

You can repeat enrollment later if recognition for a particular person needs
improvement.

Retraining one Home Assistant user replaces only that user's voice profile.
Other enrolled users are left unchanged.

---

# Testing a profile

## Test profile

After recording a sample in the sidebar panel, select **Test profile** to run it
against the trained profiles without changing enrollment.

The result shows the best candidate, similarity, runner-up margin and whether
the backend accepted the match.

The first recognition after the backend starts can be slower while the
recognition engine becomes warm. Later recognitions are normally faster.

## Live satellite test

A browser recording cannot tell you how well a real satellite performs across
a room, with its own microphone and acoustics. The **Live satellite test** is
intended for that.

1. Select the satellite in the Speaker Recognition panel.
2. Select **Start live test**.
3. Within 90 seconds, address that satellite normally using its wake word.
4. Ask a harmless question such as **“What time is it?”** or use a reversible
   command.

The normal Assist request still runs. The test only observes the exact speaker
recognition result from that turn.

The frontend reports information including:

- recognised user / candidate;
- similarity and runner-up margin;
- Home Assistant confidence threshold;
- recognition time;
- STT time;
- audio duration; and
- **added Assist latency** — the time speaker recognition actually kept Assist
  waiting after STT had already finished.

This distinction matters because recognition can take measurable CPU time while
adding no perceived delay if it finishes before STT.

---

# Recognition calibration

The panel keeps a bounded history of recent normal Assist recognition decisions
for calibration. It stores decision metadata such as scores and timing, but **not
the spoken audio or transcript**.

For recent decisions you can mark the result as:

- **Correct**;
- **Wrong speaker**; or
- **Should have recognised me**.

For wrong or missed recognitions, choose the actual Home Assistant user so the
feedback contains useful ground truth.

## Threshold guidance

After at least 15 labelled decisions, Speaker Recognition can simulate the Home
Assistant conversation confidence threshold across the collected evidence.

The recommendation deliberately treats applying the **wrong person's identity**
as more serious than failing to recognise somebody. It also distinguishes
misses that were already rejected by the backend, because changing the Home
Assistant threshold cannot rescue those cases.

A suggested threshold is **never applied automatically**. If the evidence
supports a different value, an administrator can explicitly select **Apply
suggested threshold**. The recommendation is recalculated server-side before the
setting is changed.

---

# Using Speaker Recognition with Home Assistant voice

Speaker Recognition does **not** replace your existing speech-to-text provider
or conversation agent. Instead, it wraps them with optional proxy entities.

## STT proxy

The **STT proxy** wraps an existing Home Assistant speech-to-text entity.

Your normal STT provider still converts speech into text, while Speaker
Recognition analyses the same voice input to determine who was speaking.

To add it:

1. Open **Settings → Devices & services**.
2. Open Speaker Recognition and choose **Add STT proxy**.
3. Select the speech-to-text entity you already use.

Speaker recognition and speech-to-text are performed in parallel where
possible. If speaker recognition fails, the original STT result is still
returned.

## Conversation proxy

The **Conversation proxy** wraps an existing Home Assistant conversation agent.
It consumes the recognition result correlated to that exact Assist turn.

To add it:

1. Open Speaker Recognition under **Settings → Devices & services**.
2. Choose **Add Conversation proxy**.
3. Select your existing conversation agent.
4. Set the **Minimum confidence** required before a detected speaker is used.

When the recognition result is accepted and clears the configured threshold,
the proxy can enrich a turn that does not already have a Home Assistant user.
An existing Home Assistant `Context.user_id` is preserved rather than
silently overwritten.

Your original conversation agent continues to handle the request.

> [!NOTE]
> For live satellite recognition, the Assist pipeline needs to use both the
> Speaker Recognition STT proxy and Speaker Recognition Conversation proxy.

---

# How speaker recognition works

You do not need to understand the recognition system to use the integration,
but in simplified terms:

1. You provide several recordings of a person's voice.
2. Speaker Recognition converts those recordings into mathematical voice
   embeddings.
3. It retains both a combined profile and useful information from the
   individual enrollment samples.
4. New speech is compared with the enrolled profiles.
5. A match must satisfy the backend's similarity/ambiguity checks before it can
   be considered for the additional Home Assistant confidence threshold.

The recognition engine uses
[Resemblyzer](https://github.com/resemble-ai/Resemblyzer).

Recognition confidence is useful evidence, not absolute proof of identity.

---

# Troubleshooting

## Speaker Recognition does not appear in HACS

Make sure the repository was added as a **Custom repository** with the type set
to **Integration**:

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
- the backend URL contains the correct Home Assistant/server IP address; and
- the URL begins with `http://` unless you have separately configured HTTPS.

Remote backend API requests are authenticated by default. For a standalone
backend, test the health endpoint with the same token configured in Home
Assistant:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://SPEAKER_RECOGNITION_HOST:8099/health
```

A healthy service should return JSON containing:

```json
{
  "status": "healthy"
}
```

A browser opening `/health` directly from another machine will normally receive
`401 Unauthorized`, because the browser is not sending that bearer token. The
Home Assistant OS/Supervised add-on path remains trusted through the Supervisor
gateway.

## Browser microphone recording is unavailable

Browser microphone access normally requires HTTPS (or localhost). If your Home
Assistant frontend is not in a secure browser context, use satellite enrollment
or the existing WAV selection flow instead.

## A WAV sample is rejected

The fallback WAV enrollment flow expects a local, uncompressed **16-bit PCM
WAV** file with enough usable audio. Renaming an MP3 file to `.wav` does not
convert its audio encoding.

## Recognition is unreliable

Speaker recognition is affected by recording conditions.

For better results:

- use clear recordings with little background noise;
- speak naturally during enrollment;
- avoid recordings that are extremely quiet or distorted;
- use enrollment recordings made in conditions reasonably similar to normal
  voice use;
- use **Live satellite test** to check the real microphone/path; and
- retrain the affected user if their original samples were poor.

If a missed recognition was already rejected by the backend, lowering only the
Home Assistant conversation threshold will not fix it.

---

# Updating

## Updating the integration

Updates to the Home Assistant integration are delivered through HACS.

Open **HACS** and install an available Speaker Recognition update in the same
way as other custom integrations. Restart Home Assistant if prompted.

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
  -e API_TOKEN="replace-with-a-long-random-token" \
  -v ./embeddings:/data/embeddings \
  ghcr.io/conorod1992/speaker-recognition:latest
```

Then configure the Home Assistant integration to use the address of that
server, for example:

```text
http://192.168.1.50:8099
```

Enter the same value from `API_TOKEN` in the integration's **Backend API token**
field. Remote standalone backends reject unauthenticated requests by default.
The Home Assistant OS/Supervised add-on path does not require this token because
the Supervisor gateway is trusted locally.

---

# REST API

The recognition backend also exposes a REST API for advanced integrations.
Remote callers must send `Authorization: Bearer <token>` unless insecure remote
access has been explicitly enabled. Loopback and the normal Supervisor gateway
remain trusted locally.

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
      "audio": {
        "audio_data": "base64-string",
        "sample_rate": 16000
      }
    }
  ]
}
```

Training an existing user replaces that user's recognition profile while
preserving profiles belonging to other users.

## `POST /recognize`

Attempts to identify the speaker in supplied audio.

Example request:

```json
{
  "audio": {
    "audio_data": "base64-string",
    "sample_rate": 16000
  }
}
```

A recognition result includes the candidate and decision information used by
the Home Assistant integration, including similarity, margin and whether the
backend accepted the match.

---

# Python client

The repository includes a Python client for advanced applications that need to
communicate with the Speaker Recognition service directly. This fork is **not
published to PyPI**.

For development or direct use, clone the repository and install it locally:

```bash
git clone https://github.com/conorod1992/speaker-recognition.git
cd speaker-recognition
python -m pip install .
```

To include the recognition server dependencies:

```bash
python -m pip install ".[server]"
```

The package currently supports Python 3.9.

Release builds also attach the source archive and wheel to the corresponding
GitHub Release for advanced users who prefer a packaged artifact.

Example recognition request against an authenticated remote backend:

```python
from speaker_recognition import SpeakerRecognitionClient
from speaker_recognition.models import AudioInput, RecognitionRequest

async with SpeakerRecognitionClient(
    "http://192.168.1.50:8099",
    api_token="replace-with-the-backend-token",
) as client:
    result = await client.recognize(
        RecognitionRequest(
            audio=AudioInput(
                audio_data="<base64-encoded-audio>",
                sample_rate=16000,
            )
        )
    )

    print(
        f"Speaker: {result.user_id or result.candidate_user_id} "
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
| `API_TOKEN` | Bearer token for non-local API clients | empty |
| `ALLOW_INSECURE_REMOTE` | Permit unauthenticated non-local clients | `false` |

---

# Development

## Requirements

For server development:

- Python 3.9
- [uv](https://github.com/astral-sh/uv)

The packaged backend/client currently supports Python 3.9.

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
- add or update tests for behavioural changes; and
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

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgements

Speaker Recognition makes use of:

- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) for speaker voice
  embeddings
- [Home Assistant](https://www.home-assistant.io/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

Made with ❤️ for the Home Assistant community

### Advanced recognition thresholds

In the panel's **Settings → Advanced recognition thresholds**, administrators can
set minimum accepted similarity (default **0.55**) and minimum accepted margin
(default **0.05**). Both accept values from 0 to 1. A margin of 0 disables the
ambiguity gate; similarity is still required. A single enrolled profile has no
competing margin. Lowering either threshold can increase incorrect identities.
These backend gates are separate from the Conversation proxy's minimum identity
confidence, which continues to govern whether an accepted identity is applied.

The existing HA config entry persists the policy. HA sends it with each
recognition request and the authoritative backend applies and echoes it. This
avoids separate settings files, shared-backend global mutations, extra round trips,
and restart windows with stale defaults. Settings show the policy used by this HA
entry; `/health` exposes the backend default policy and capability, and recognition
responses expose the applied policy. Existing entries without a policy keep their
original requests and exact defaults. After explicitly changing thresholds, upgrade
the backend too: if it cannot acknowledge the policy, Assist continues without
speaker identity rather than silently using a different policy. No shadow-engine,
DSP, profile learning, enrollment or correlation behaviour changes.

### Joint backend calibration

Threshold guidance also evaluates the backend's similarity and margin gates
against the existing bounded labelled history, including rejected decisions.
It requires at least 15 labelled decisions with usable raw similarity, margin and
candidate evidence. Missing old evidence is skipped, never guessed from confidence.
A fixed grid at 0.05 increments (plus the exact current policy) has at most 484
pairs. It counts correct identities, wrong identities, enrolled-speaker misses
and correctly rejected unknowns, with wrong identities costing five times a miss.
Ties favour fewer wrong identities, then proximity to the current policy, then
stricter gates. This is guidance from your labels, not a guarantee of future accuracy.

**Apply recommended backend thresholds** is an explicit administrator action.
The server recalculates from the latest history immediately before saving to the
existing config entry; the browser cannot submit a cached threshold recommendation.
Nothing is applied automatically. Conversation proxy confidence calibration remains
separate and unchanged. No audio is retained beyond the existing bounded review
queue for this analysis, and no model inference is performed.

### Improve a profile with reviewed Assist clips

After explicitly labelling a retained calibration clip for an active enrolled HA
user, choose **Add to voice profile**. A confirmed "Correct" identity is an explicit
label; a prediction alone is never sufficient. Unknown/not-enrolled speakers and
expired audio cannot be promoted. The target comes from the label, even when the
predicted candidate was someone else. Each decision can be promoted only once.

Promotion copies audio to integration-managed enrollment media and stages it;
it does not train immediately. In Enrollment, select the user and choose
**Train with promoted clips** to append the clips to existing training recordings,
or finish a full set of replacement phrases to train those together with the clips.
The old profile stays active until transactional training succeeds. Failed training
preserves the previous profile and the staged clips for retry. **Discard promoted
clips** removes pending managed audio. Removing a profile cleans its pending media;
replacement cleanup preserves any recordings still referenced by the new profile.

The staged media index uses a small versioned Home Assistant Store so pending clips
survive restart without entering the live voice-sample configuration prematurely.
It holds at most 24 pending clips overall, with at most six promoted clips per user
across pending and current training material. Only explicit promotions gain this
managed training lifetime; the ten-clip rolling review-audio queue and 200-record
history limits remain unchanged. No automatic self-training is performed.

### Advisory enrollment embedding checks

Each captured browser or satellite enrollment recording is saved first, then
analysed asynchronously by the authoritative local embedding engine. Once three
staged samples exist, the panel shows **Good sample**, **Inconsistent sample — retake
recommended**, or **Insufficient evidence yet**. Details show internal consistency
and, when available, similarity to that user's existing profile. Preview never
writes a trained profile. Failure or an older backend without the preview endpoint
leaves the recording usable and shows analysis unavailable.

The preview reuses training's normalization and outlier diagnostics. An additional
average-consistency cutoff of 0.5 is only a conservative advisory heuristic, not a
universal speaker-quality boundary or a new training rejection rule. Final training
still performs its existing minimum-sample, outlier and transactional checks.
Successful preview embeddings are cached only in backend memory, keyed by engine,
sample rate and audio hash, for up to 15 minutes and at most 48 samples. Training
can reuse those embeddings while rerunning all profile checks. No cache is used on
normal Assist recognition and no audio is retained in this cache.

Analysis is on capture only, coalesces rapid retakes, ignores stale results, and is
cancelled on integration unload. Each request handles at most six phrase samples plus six explicitly promoted clips.
Promoted Assist clips are labelled separately and analysed when promoted, too.
It uses the backend's existing authoritative lock; busy analysis degrades gracefully
and Assist never queues behind it. There are no continuous background ML jobs.

### Cross-speaker profile health

The panel's **Diagnostics → Profiles → Profile health** compares already stored
embeddings. Refresh it after profile changes to see internal consistency, the
nearest other enrolled HA user, and separation (`1 - cosine similarity`, from
0 to 2; higher means farther apart). One-user installations show “No other
enrolled speakers to compare”. Missing legacy sample embeddings leave centroid
comparisons available and mark sample diagnostics incomplete.

A separation below 0.10 prompts a cautious review warning. Stored samples are
flagged when another profile's similarity is at least 0.80 and the sample's
own-versus-other similarity gap is at most 0.05. These are conservative review
heuristics, not scientifically universal cutoffs or additional recognition gates.
Sample numbers refer to retained backend embeddings, not original phrase slots
(which can differ after training removes outliers). Review recordings before
choosing whether to retrain; nothing is changed automatically.

Diagnostics use a short, consistent snapshot of existing profile/sample embeddings;
there is no additional model inference, background ML job, or persisted derived
state. Retraining and removing profiles are reflected on the next refresh. The
admin-only HA endpoint resolves active eligible users to their display names.
An older or temporarily unavailable backend leaves profile health unavailable
without affecting recognition or enrollment.
