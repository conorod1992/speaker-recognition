const TARGET_SAMPLE_RATE = 16000;

class SpeakerRecognitionPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._info = null;
    this._loading = true;
    this._error = null;
    this._selectedUser = "";
    this._samples = [];
    this._sampleIndex = 0;
    this._pendingSample = null;
    this._trainingResult = null;
    this._testResult = null;
    this._recording = false;
    this._recordMode = null;
    this._audioContext = null;
    this._mediaStream = null;
    this._processor = null;
    this._source = null;
    this._silentGain = null;
    this._recordedChunks = [];
    this._recordStartedAt = 0;
    this._autoStopTimer = null;
    this._playbackUrl = null;
  }

  set hass(value) {
    this._hass = value;
    if (this.isConnected && !this._info && !this._loadingPromise) {
      this._load();
    }
  }

  set panel(value) {
    this._panel = value;
  }

  set narrow(value) {
    this._narrow = value;
  }

  set route(value) {
    this._route = value;
  }

  connectedCallback() {
    this._render();
    if (this._hass && !this._info && !this._loadingPromise) {
      this._load();
    }
  }

  disconnectedCallback() {
    this._cleanupRecording();
    this._clearPlaybackUrl();
  }

  async _call(type, data = {}) {
    if (!this._hass) {
      throw new Error("Home Assistant connection is not ready");
    }
    return this._hass.connection.sendMessagePromise({ type, ...data });
  }

  async _load() {
    this._loading = true;
    this._error = null;
    this._render();
    this._loadingPromise = this._call("speaker_recognition/enrollment/info");
    try {
      this._info = await this._loadingPromise;
      if (!this._selectedUser && this._info.users.length) {
        this._selectedUser = this._info.users[0].id;
      }
    } catch (error) {
      this._error = this._errorText(error);
    } finally {
      this._loading = false;
      this._loadingPromise = null;
      this._render();
    }
  }

  _microphoneAvailable() {
    return Boolean(navigator.mediaDevices?.getUserMedia);
  }

  _errorText(error) {
    if (!error) return "Unknown error";
    if (typeof error === "string") return error;
    return error.message || error.error?.message || String(error);
  }

  _escape(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  _render() {
    if (!this.shadowRoot) return;

    const micAvailable = this._microphoneAvailable();
    const users = this._info?.users || [];
    const enrolled = new Set(this._info?.enrolled_users || []);
    const maxSamples = this._info?.maximum_samples || 6;
    const minSamples = this._info?.minimum_samples || 5;
    const currentPhrase = this._info?.phrases?.[this._sampleIndex] || "";

    let body = "";
    if (this._loading) {
      body = `<div class="status">Loading speaker profiles…</div>`;
    } else if (this._error) {
      body = `
        <div class="notice error">${this._escape(this._error)}</div>
        <button class="secondary" id="retry-load">Retry</button>
      `;
    } else if (!users.length) {
      body = `<div class="notice error">No active Home Assistant users are available for enrollment.</div>`;
    } else {
      const userOptions = users
        .map(
          (user) => `<option value="${this._escape(user.id)}" ${
            user.id === this._selectedUser ? "selected" : ""
          }>${this._escape(user.name)}${enrolled.has(user.id) ? " · enrolled" : ""}</option>`,
        )
        .join("");

      body = `
        <section class="card">
          <h2>Voice profile</h2>
          <p class="muted">Choose the Home Assistant user whose voice you want to enroll or retrain.</p>
          <label for="user-select">User</label>
          <select id="user-select">${userOptions}</select>
          <div class="profile-state">${
            enrolled.has(this._selectedUser)
              ? "A persisted profile is already available. Completing enrollment will replace it."
              : "No persisted voice profile is currently reported for this user."
          }</div>
        </section>

        <section class="card">
          <h2>Record phrases</h2>
          ${
            micAvailable
              ? this._renderRecorder(currentPhrase, minSamples, maxSamples)
              : `<div class="notice warning">
                  Microphone capture is not available in this Home Assistant view. Browser microphone access normally requires a secure HTTPS context. You can open Home Assistant through an HTTPS URL (including Home Assistant Cloud remote access where configured), try the Companion app if its WebView exposes microphone access, or use the existing WAV upload enrollment flow.
                </div>
                <a class="link-button" href="/config/integrations/integration/speaker_recognition">Open Speaker Recognition settings</a>`
          }
        </section>

        <section class="card">
          <h2>Test profile</h2>
          <p class="muted">Record a fresh sentence that was not used for enrollment. Test audio is not stored.</p>
          ${
            micAvailable && enrolled.has(this._selectedUser)
              ? `<button class="secondary" id="test-record" ${this._recording ? "disabled" : ""}>Record a test sample</button>`
              : micAvailable
                ? `<div class="muted">Complete enrollment first to test this profile.</div>`
                : ""
          }
          ${this._renderTestResult()}
        </section>

        <section class="card privacy">
          <h2>What is stored?</h2>
          <p>Your recordings are sent through your authenticated Home Assistant connection to the configured Speaker Recognition backend for training. The browser recordings are not saved by this panel. The backend persists voice embeddings, not the raw microphone audio.</p>
        </section>
      `;
    }

    this.shadowRoot.innerHTML = `
      <style>${this._styles()}</style>
      <div class="page">
        <header>
          <div>
            <h1>Speaker Recognition</h1>
            <p>Guided microphone enrollment and profile testing</p>
          </div>
        </header>
        ${body}
      </div>
    `;

    this._bindEvents();
  }

  _renderRecorder(phrase, minSamples, maxSamples) {
    if (this._trainingResult) {
      const consistency = this._trainingResult.profile_consistency;
      const outliers = this._trainingResult.outlier_samples || [];
      return `
        <div class="notice success">
          <strong>Profile trained successfully.</strong>
          ${
            typeof consistency === "number"
              ? `<div>Enrollment consistency: <strong>${consistency.toFixed(3)}</strong></div>`
              : ""
          }
          ${
            outliers.length
              ? `<div>Samples worth retrying: ${outliers.join(", ")}</div>`
              : `<div>No strong enrollment outliers were detected.</div>`
          }
        </div>
        <button class="secondary" id="start-over">Enroll or retrain again</button>
      `;
    }

    if (this._pendingSample) {
      const warnings = this._pendingSample.quality.warnings;
      return `
        <div class="progress">Sample ${this._sampleIndex + 1} of ${maxSamples}</div>
        <blockquote>${this._escape(phrase)}</blockquote>
        <audio controls src="${this._escape(this._pendingSample.playbackUrl)}"></audio>
        <div class="quality ${warnings.length ? "quality-warning" : "quality-good"}">
          <strong>${warnings.length ? "Recording captured with a warning" : "Good recording"}</strong>
          <span>${this._pendingSample.quality.duration.toFixed(1)} s · peak ${(this._pendingSample.quality.peak * 100).toFixed(0)}%</span>
          ${warnings.map((warning) => `<span>${this._escape(warning)}</span>`).join("")}
        </div>
        <div class="actions">
          <button class="primary" id="accept-sample">Use recording</button>
          <button class="secondary" id="retry-sample">Retry</button>
        </div>
      `;
    }

    if (this._recording && this._recordMode === "enroll") {
      return `
        <div class="progress">Sample ${this._sampleIndex + 1} of ${maxSamples}</div>
        <blockquote>${this._escape(phrase)}</blockquote>
        <div class="recording"><span class="record-dot"></span> Recording… speak naturally, then stop.</div>
        <button class="danger" id="stop-record">Stop recording</button>
      `;
    }

    if (this._samples.length >= minSamples) {
      return `
        <div class="progress">${this._samples.length} recordings accepted</div>
        <blockquote>${this._escape(phrase)}</blockquote>
        <p class="muted">You already have enough samples. You can train now, or record the optional final phrase for a little more voice variation.</p>
        <div class="actions">
          <button class="primary" id="train-profile">Train profile now</button>
          ${
            this._samples.length < maxSamples
              ? `<button class="secondary" id="record-sample">Record optional sample ${this._sampleIndex + 1}</button>`
              : ""
          }
          <button class="text-button" id="reset-samples">Start over</button>
        </div>
      `;
    }

    return `
      <div class="progress">Sample ${this._sampleIndex + 1} of ${maxSamples} · ${this._samples.length} accepted</div>
      <blockquote>${this._escape(phrase)}</blockquote>
      <p class="muted">Press record, say the phrase naturally, then stop. Five good recordings are required; the sixth is optional.</p>
      <button class="primary" id="record-sample" ${this._recording ? "disabled" : ""}>Start recording</button>
      ${this._samples.length ? `<button class="text-button" id="reset-samples">Start over</button>` : ""}
    `;
  }

  _renderTestResult() {
    if (this._recording && this._recordMode === "test") {
      return `
        <div class="recording"><span class="record-dot"></span> Recording test sample…</div>
        <button class="danger" id="stop-record">Stop recording</button>
      `;
    }
    if (!this._testResult) return "";
    if (this._testResult.error) {
      return `<div class="notice error">${this._escape(this._testResult.error)}</div>`;
    }
    const margin = this._testResult.margin;
    const scores = Object.entries(this._testResult.all_scores || {})
      .sort((a, b) => b[1] - a[1])
      .map(([user, score]) => `<li>${this._escape(this._userName(user))}: ${Number(score).toFixed(3)}</li>`)
      .join("");
    return `
      <div class="test-result ${this._testResult.accepted ? "accepted" : "rejected"}">
        <strong>${this._testResult.accepted ? "Accepted match" : "Unknown / ambiguous speaker"}</strong>
        <div>Candidate: ${this._escape(this._userName(this._testResult.candidate_user_id))}</div>
        <div>Similarity: ${Number(this._testResult.similarity).toFixed(3)}</div>
        <div>Runner-up margin: ${margin == null ? "n/a" : Number(margin).toFixed(3)}</div>
        <ul>${scores}</ul>
      </div>
    `;
  }

  _userName(userId) {
    return this._info?.users?.find((user) => user.id === userId)?.name || userId || "Unknown";
  }

  _bindEvents() {
    const root = this.shadowRoot;
    if (!root) return;

    root.getElementById("retry-load")?.addEventListener("click", () => this._load());
    root.getElementById("user-select")?.addEventListener("change", (event) => {
      this._selectedUser = event.target.value;
      this._resetEnrollmentState();
      this._testResult = null;
      this._render();
    });
    root.getElementById("record-sample")?.addEventListener("click", () => this._startRecording("enroll"));
    root.getElementById("test-record")?.addEventListener("click", () => this._startRecording("test"));
    root.getElementById("stop-record")?.addEventListener("click", () => this._stopRecording());
    root.getElementById("accept-sample")?.addEventListener("click", () => this._acceptSample());
    root.getElementById("retry-sample")?.addEventListener("click", () => {
      this._discardPendingSample();
      this._render();
    });
    root.getElementById("train-profile")?.addEventListener("click", () => this._trainProfile());
    root.getElementById("reset-samples")?.addEventListener("click", () => {
      this._resetEnrollmentState();
      this._render();
    });
    root.getElementById("start-over")?.addEventListener("click", () => {
      this._resetEnrollmentState();
      this._render();
    });
  }

  async _startRecording(mode) {
    this._error = null;
    this._testResult = mode === "test" ? null : this._testResult;
    if (!this._microphoneAvailable()) {
      this._error = "Microphone access is not available in this Home Assistant view.";
      this._render();
      return;
    }

    try {
      this._mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      });
      this._audioContext = new AudioContext();
      this._source = this._audioContext.createMediaStreamSource(this._mediaStream);
      this._processor = this._audioContext.createScriptProcessor(4096, 1, 1);
      this._silentGain = this._audioContext.createGain();
      this._silentGain.gain.value = 0;
      this._recordedChunks = [];
      this._processor.onaudioprocess = (event) => {
        if (!this._recording) return;
        const channel = event.inputBuffer.getChannelData(0);
        this._recordedChunks.push(new Float32Array(channel));
      };
      this._source.connect(this._processor);
      this._processor.connect(this._silentGain);
      this._silentGain.connect(this._audioContext.destination);
      this._recording = true;
      this._recordMode = mode;
      this._recordStartedAt = performance.now();
      const maximumSeconds = this._info?.maximum_seconds || 10;
      this._autoStopTimer = window.setTimeout(() => this._stopRecording(), maximumSeconds * 1000);
      this._render();
    } catch (error) {
      this._cleanupRecording();
      this._error = `Unable to access the microphone: ${this._errorText(error)}`;
      this._render();
    }
  }

  async _stopRecording() {
    if (!this._recording || !this._audioContext) return;
    const mode = this._recordMode;
    const inputRate = this._audioContext.sampleRate;
    const chunks = this._recordedChunks;
    this._recording = false;
    this._cleanupRecording();

    const merged = this._mergeChunks(chunks);
    const resampled = this._resample(merged, inputRate, TARGET_SAMPLE_RATE);
    const pcm = this._floatToPcm16(resampled);
    const quality = this._quality(resampled);

    if (quality.duration < (this._info?.minimum_seconds || 0.5)) {
      this._error = "That recording was too short. Please record the full phrase.";
      this._render();
      return;
    }

    if (mode === "test") {
      await this._submitTest(pcm);
      return;
    }

    this._clearPlaybackUrl();
    const wav = this._pcmToWav(pcm, TARGET_SAMPLE_RATE);
    this._playbackUrl = URL.createObjectURL(wav);
    this._pendingSample = {
      pcm,
      base64: this._bytesToBase64(new Uint8Array(pcm.buffer)),
      quality,
      playbackUrl: this._playbackUrl,
    };
    this._render();
  }

  _cleanupRecording() {
    this._recording = false;
    if (this._autoStopTimer) {
      clearTimeout(this._autoStopTimer);
      this._autoStopTimer = null;
    }
    if (this._processor) {
      this._processor.onaudioprocess = null;
      try { this._processor.disconnect(); } catch (_) {}
    }
    if (this._source) {
      try { this._source.disconnect(); } catch (_) {}
    }
    if (this._silentGain) {
      try { this._silentGain.disconnect(); } catch (_) {}
    }
    if (this._mediaStream) {
      this._mediaStream.getTracks().forEach((track) => track.stop());
    }
    if (this._audioContext && this._audioContext.state !== "closed") {
      this._audioContext.close().catch(() => {});
    }
    this._processor = null;
    this._source = null;
    this._silentGain = null;
    this._mediaStream = null;
    this._audioContext = null;
    this._recordedChunks = [];
    this._recordMode = null;
  }

  _mergeChunks(chunks) {
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }

  _resample(input, inputRate, outputRate) {
    if (inputRate === outputRate) return input;
    if (!input.length) return new Float32Array();
    const ratio = inputRate / outputRate;
    const outputLength = Math.max(1, Math.round(input.length / ratio));
    const output = new Float32Array(outputLength);
    for (let i = 0; i < outputLength; i += 1) {
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const fraction = position - left;
      output[i] = input[left] * (1 - fraction) + input[right] * fraction;
    }
    return output;
  }

  _floatToPcm16(input) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const value = Math.max(-1, Math.min(1, input[i]));
      output[i] = value < 0 ? Math.round(value * 32768) : Math.round(value * 32767);
    }
    return output;
  }

  _quality(input) {
    let peak = 0;
    let clipped = 0;
    for (const value of input) {
      const absolute = Math.abs(value);
      if (absolute > peak) peak = absolute;
      if (absolute >= 0.98) clipped += 1;
    }
    const duration = input.length / TARGET_SAMPLE_RATE;
    const clippingRatio = input.length ? clipped / input.length : 0;
    const warnings = [];
    if (peak < 0.025) warnings.push("Your voice was very quiet. Moving closer to the microphone may improve the profile.");
    if (clippingRatio > 0.01) warnings.push("The recording clipped frequently. Moving slightly farther from the microphone may help.");
    if (duration < 1.0) warnings.push("This is a very short sample. Make sure the full phrase was captured.");
    return { duration, peak, clippingRatio, warnings };
  }

  _acceptSample() {
    if (!this._pendingSample) return;
    this._samples.push(this._pendingSample.base64);
    this._discardPendingSample();
    this._sampleIndex = this._samples.length;
    const maxSamples = this._info?.maximum_samples || 6;
    if (this._samples.length >= maxSamples) {
      this._trainProfile();
      return;
    }
    this._render();
  }

  _discardPendingSample() {
    this._pendingSample = null;
    this._clearPlaybackUrl();
  }

  _clearPlaybackUrl() {
    if (this._playbackUrl) {
      URL.revokeObjectURL(this._playbackUrl);
      this._playbackUrl = null;
    }
  }

  _resetEnrollmentState() {
    this._cleanupRecording();
    this._discardPendingSample();
    this._samples = [];
    this._sampleIndex = 0;
    this._trainingResult = null;
    this._error = null;
  }

  async _trainProfile() {
    const minimum = this._info?.minimum_samples || 5;
    if (!this._selectedUser || this._samples.length < minimum) return;
    this._loading = true;
    this._error = null;
    this._render();
    try {
      this._trainingResult = await this._call("speaker_recognition/enrollment/train", {
        user_id: this._selectedUser,
        sample_rate: TARGET_SAMPLE_RATE,
        samples: this._samples,
      });
      this._info = await this._call("speaker_recognition/enrollment/info");
    } catch (error) {
      this._error = `Training failed: ${this._errorText(error)}`;
    } finally {
      this._loading = false;
      this._render();
    }
  }

  async _submitTest(pcm) {
    this._loading = true;
    this._render();
    try {
      this._testResult = await this._call("speaker_recognition/enrollment/test", {
        sample_rate: TARGET_SAMPLE_RATE,
        audio_data: this._bytesToBase64(new Uint8Array(pcm.buffer)),
      });
    } catch (error) {
      this._testResult = { error: this._errorText(error) };
    } finally {
      this._loading = false;
      this._render();
    }
  }

  _bytesToBase64(bytes) {
    const chunkSize = 0x8000;
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += chunkSize) {
      const chunk = bytes.subarray(offset, Math.min(offset + chunkSize, bytes.length));
      binary += String.fromCharCode(...chunk);
    }
    return btoa(binary);
  }

  _pcmToWav(pcm, sampleRate) {
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const write = (offset, text) => {
      for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
    };
    write(0, "RIFF");
    view.setUint32(4, 36 + pcm.length * 2, true);
    write(8, "WAVE");
    write(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    write(36, "data");
    view.setUint32(40, pcm.length * 2, true);
    for (let i = 0; i < pcm.length; i += 1) view.setInt16(44 + i * 2, pcm[i], true);
    return new Blob([buffer], { type: "audio/wav" });
  }

  _styles() {
    return `
      :host {
        display: block;
        min-height: 100%;
        background: var(--primary-background-color);
        color: var(--primary-text-color);
        font-family: var(--paper-font-body1_-_font-family, system-ui, sans-serif);
      }
      * { box-sizing: border-box; }
      .page { max-width: 880px; margin: 0 auto; padding: 24px 20px 48px; }
      header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
      h1 { font-size: 28px; margin: 0 0 4px; }
      header p, .muted { color: var(--secondary-text-color); }
      header p { margin: 0; }
      .card {
        background: var(--card-background-color);
        border-radius: var(--ha-card-border-radius, 12px);
        box-shadow: var(--ha-card-box-shadow, 0 2px 8px rgba(0,0,0,.12));
        padding: 20px;
        margin: 16px 0;
      }
      h2 { margin: 0 0 10px; font-size: 20px; }
      p { line-height: 1.5; }
      label { display: block; font-weight: 600; margin: 16px 0 6px; }
      select {
        width: 100%; padding: 11px 12px; border-radius: 8px;
        border: 1px solid var(--divider-color); background: var(--card-background-color);
        color: var(--primary-text-color); font-size: 15px;
      }
      .profile-state, .progress { margin-top: 12px; color: var(--secondary-text-color); font-size: 14px; }
      blockquote {
        margin: 16px 0; padding: 16px 18px; border-left: 4px solid var(--primary-color);
        background: var(--secondary-background-color); border-radius: 6px; font-size: 18px; line-height: 1.45;
      }
      button, .link-button {
        appearance: none; border: 0; border-radius: 8px; padding: 10px 16px;
        font-weight: 600; font-size: 14px; cursor: pointer; text-decoration: none;
        display: inline-flex; align-items: center; justify-content: center; margin-top: 10px;
      }
      button:disabled { opacity: .5; cursor: default; }
      .primary, .link-button { background: var(--primary-color); color: var(--text-primary-color, white); }
      .secondary { background: var(--secondary-background-color); color: var(--primary-text-color); }
      .danger { background: var(--error-color, #db4437); color: white; }
      .text-button { background: transparent; color: var(--primary-color); }
      .actions { display: flex; flex-wrap: wrap; gap: 8px; }
      .notice { padding: 14px 16px; border-radius: 8px; margin: 12px 0; line-height: 1.45; }
      .notice.error { background: color-mix(in srgb, var(--error-color, #db4437) 14%, transparent); }
      .notice.warning { background: color-mix(in srgb, var(--warning-color, #f9a825) 18%, transparent); }
      .notice.success { background: color-mix(in srgb, var(--success-color, #43a047) 14%, transparent); }
      .recording { margin: 16px 0 4px; font-weight: 600; }
      .record-dot { width: 10px; height: 10px; border-radius: 50%; background: var(--error-color, #db4437); display: inline-block; margin-right: 7px; animation: pulse 1.2s infinite; }
      @keyframes pulse { 50% { opacity: .35; } }
      audio { width: 100%; margin: 8px 0 12px; }
      .quality { display: grid; gap: 4px; padding: 12px 14px; border-radius: 8px; margin: 8px 0; }
      .quality-good { background: color-mix(in srgb, var(--success-color, #43a047) 12%, transparent); }
      .quality-warning { background: color-mix(in srgb, var(--warning-color, #f9a825) 14%, transparent); }
      .quality span { font-size: 14px; }
      .test-result { margin-top: 14px; padding: 14px; border-radius: 8px; line-height: 1.45; }
      .test-result.accepted { background: color-mix(in srgb, var(--success-color, #43a047) 12%, transparent); }
      .test-result.rejected { background: color-mix(in srgb, var(--warning-color, #f9a825) 14%, transparent); }
      .test-result ul { margin: 8px 0 0; padding-left: 20px; }
      .privacy p { margin-bottom: 0; }
      .status { padding: 30px 0; color: var(--secondary-text-color); }
      @media (max-width: 600px) {
        .page { padding: 16px 12px 36px; }
        .card { padding: 16px; }
        h1 { font-size: 24px; }
        blockquote { font-size: 16px; }
        .actions { flex-direction: column; align-items: stretch; }
        .actions button { width: 100%; }
      }
    `;
  }
}

customElements.define("speaker-recognition-panel", SpeakerRecognitionPanel);
