class SpeakerRecognitionPanel extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: "open" });
    this._hass = null;
    this._status = null;
    this._userId = "";
    this._sampleIndex = 0;
    this._satelliteId = "";
    this._recording = null;
    this._lastWav = null;
    this._message = "";
    this._busy = false;
    this._pollTimer = null;
  }

  set hass(value) {
    this._hass = value;
    if (!this._status) this._refresh();
  }

  connectedCallback() {
    this._render();
  }

  disconnectedCallback() {
    if (this._pollTimer) clearTimeout(this._pollTimer);
    this._stopRecorderTracks();
  }

  async _call(message) {
    if (!this._hass) throw new Error("Home Assistant connection is not ready");
    return this._hass.connection.sendMessagePromise(message);
  }

  async _refresh(silent = false) {
    if (!this._hass) return;
    try {
      this._status = await this._call({ type: "speaker_recognition/status" });
      if (!this._userId && this._status.users.length) this._userId = this._status.users[0].id;
      if (!this._satelliteId && this._status.satellites.length) {
        this._satelliteId = this._status.satellites[0].entity_id;
      }
      const staged = (this._status.staged && this._status.staged[this._userId]) || [];
      if (staged.length && !staged.includes(this._sampleIndex)) {
        const next = Math.min(staged.length, this._status.phrases.length - 1);
        this._sampleIndex = next;
      }
      if (!silent) this._message = "";
    } catch (err) {
      this._message = this._errorText(err);
    }
    this._render();
  }

  _errorText(err) {
    return err && (err.message || err.code) ? String(err.message || err.code) : String(err);
  }

  _stagedIndexes() {
    if (!this._status || !this._status.staged) return [];
    return this._status.staged[this._userId] || [];
  }

  _canUseMicrophone() {
    return Boolean(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  async _startRecording() {
    if (!this._canUseMicrophone()) {
      this._message = "Microphone recording needs a secure browser context. Open Home Assistant over HTTPS, or use the upload/satellite alternatives.";
      this._render();
      return;
    }
    try {
      this._message = "";
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(4096, 1, 1);
      const chunks = [];
      processor.onaudioprocess = (event) => {
        chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      source.connect(processor);
      processor.connect(context.destination);
      this._recording = { stream, context, source, processor, chunks, sampleRate: context.sampleRate };
      this._render();
    } catch (err) {
      this._message = `Could not access the microphone: ${this._errorText(err)}`;
      this._render();
    }
  }

  _stopRecorderTracks() {
    const rec = this._recording;
    if (!rec) return;
    try { rec.processor.disconnect(); } catch (_) {}
    try { rec.source.disconnect(); } catch (_) {}
    for (const track of rec.stream.getTracks()) track.stop();
    rec.context.close().catch(() => {});
  }

  async _stopRecording() {
    const rec = this._recording;
    if (!rec) return;
    this._stopRecorderTracks();
    const length = rec.chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(length);
    let offset = 0;
    for (const chunk of rec.chunks) {
      samples.set(chunk, offset);
      offset += chunk.length;
    }
    this._recording = null;
    if (samples.length < rec.sampleRate / 2) {
      this._message = "That recording was too short. Please try the phrase again.";
      this._render();
      return;
    }
    this._lastWav = this._encodeWav(samples, rec.sampleRate);
    this._render();
  }

  _encodeWav(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const write = (offset, text) => {
      for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    };
    write(0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
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
    view.setUint32(40, samples.length * 2, true);
    let pos = 44;
    for (let i = 0; i < samples.length; i++, pos += 2) {
      const value = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(pos, value < 0 ? value * 32768 : value * 32767, true);
    }
    return new Uint8Array(buffer);
  }

  _bytesToBase64(bytes) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
  }

  _playRecording() {
    if (!this._lastWav) return;
    const url = URL.createObjectURL(new Blob([this._lastWav], { type: "audio/wav" }));
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    audio.play();
  }

  async _useRecording() {
    if (!this._lastWav || !this._userId) return;
    this._busy = true;
    this._message = "Saving sample…";
    this._render();
    try {
      const quality = await this._call({
        type: "speaker_recognition/stage_sample",
        user_id: this._userId,
        sample_index: this._sampleIndex,
        wav_base64: this._bytesToBase64(this._lastWav),
      });
      const warnings = [];
      if (quality.too_quiet) warnings.push("audio level was low");
      if (quality.clipping) warnings.push("some audio clipped");
      this._message = `Sample ${this._sampleIndex + 1} saved (${quality.duration}s)` +
        (warnings.length ? `. Note: ${warnings.join(" and ")}.` : ".");
      this._lastWav = null;
      await this._refresh(true);
      if (this._sampleIndex < this._status.phrases.length - 1) this._sampleIndex += 1;
    } catch (err) {
      this._message = this._errorText(err);
    } finally {
      this._busy = false;
      this._render();
    }
  }

  async _startSatellite() {
    if (!this._userId || !this._satelliteId) return;
    this._busy = true;
    this._message = "The selected satellite will read the phrase and listen for your reply…";
    this._render();
    try {
      const started = await this._call({
        type: "speaker_recognition/start_satellite_sample",
        user_id: this._userId,
        satellite_id: this._satelliteId,
        sample_index: this._sampleIndex,
      });
      this._pollForSatelliteCapture(this._sampleIndex, started.session_id);
    } catch (err) {
      this._busy = false;
      this._message = this._errorText(err);
      this._render();
    }
  }

  _pollForSatelliteCapture(index, sessionId) {
    if (this._pollTimer) clearTimeout(this._pollTimer);
    const started = Date.now();
    const poll = async () => {
      await this._refresh(true);
      const completed = this._status.completed_satellite_captures || [];
      if (completed.includes(sessionId)) {
        this._busy = false;
        this._message = `Sample ${index + 1} captured from the satellite.`;
        if (this._sampleIndex < this._status.phrases.length - 1) this._sampleIndex += 1;
        this._render();
        return;
      }
      if (Date.now() - started > 90000) {
        this._busy = false;
        this._message = "No matching satellite utterance was captured. Please try again.";
        this._render();
        return;
      }
      this._pollTimer = setTimeout(poll, 1000);
    };
    this._pollTimer = setTimeout(poll, 1000);
  }

  async _commitEnrollment() {
    this._busy = true;
    this._message = "Training the updated profile…";
    this._render();
    try {
      const result = await this._call({
        type: "speaker_recognition/commit_enrollment",
        user_id: this._userId,
      });
      this._message = `${result.samples} samples committed. Home Assistant is applying the new profile transactionally.`;
      setTimeout(() => this._refresh(true), 1500);
    } catch (err) {
      this._message = this._errorText(err);
    } finally {
      this._busy = false;
      this._render();
    }
  }

  async _testProfile() {
    if (!this._lastWav) {
      this._message = "Record an arbitrary phrase first, then choose Test profile.";
      this._render();
      return;
    }
    this._busy = true;
    this._render();
    try {
      const result = await this._call({
        type: "speaker_recognition/test_sample",
        wav_base64: this._bytesToBase64(this._lastWav),
      });
      if (!result.available) {
        this._message = "No trained speaker profile is currently available.";
      } else {
        const similarity = Number(result.similarity).toFixed(3);
        const margin = result.margin == null ? "n/a" : Number(result.margin).toFixed(3);
        this._message = `Candidate: ${result.candidate_user_id || "unknown"} · similarity ${similarity} · margin ${margin} · ${result.accepted ? "accepted" : "unknown/rejected"}`;
      }
    } catch (err) {
      this._message = this._errorText(err);
    } finally {
      this._busy = false;
      this._render();
    }
  }

  _render() {
    if (!this.shadowRoot) return;
    const s = this._status;
    const phrase = s && s.phrases[this._sampleIndex] ? s.phrases[this._sampleIndex] : "";
    const staged = this._stagedIndexes();
    const minimum = s ? s.minimum_samples : 5;
    const micAvailable = this._canUseMicrophone();
    this.shadowRoot.innerHTML = `
      <style>
        :host { display:block; box-sizing:border-box; padding:24px; color:var(--primary-text-color); background:var(--primary-background-color); min-height:100vh; }
        * { box-sizing:border-box; }
        .wrap { max-width:900px; margin:0 auto; }
        h1 { margin-top:0; font-size:28px; }
        h2 { margin:0 0 12px; font-size:20px; }
        .card { background:var(--card-background-color); border-radius:12px; padding:20px; margin:16px 0; box-shadow:var(--ha-card-box-shadow, 0 2px 6px rgba(0,0,0,.15)); }
        .row { display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
        label { display:block; font-weight:600; margin:10px 0 6px; }
        select, button { font:inherit; }
        select { min-width:260px; padding:9px; border-radius:6px; background:var(--card-background-color); color:var(--primary-text-color); border:1px solid var(--divider-color); }
        button { padding:10px 15px; border:0; border-radius:8px; cursor:pointer; background:var(--primary-color); color:var(--text-primary-color, white); }
        button.secondary { background:var(--secondary-background-color); color:var(--primary-text-color); border:1px solid var(--divider-color); }
        button:disabled { opacity:.5; cursor:not-allowed; }
        .phrase { font-size:20px; line-height:1.45; padding:16px; border-left:4px solid var(--primary-color); background:var(--secondary-background-color); border-radius:6px; }
        .muted { color:var(--secondary-text-color); }
        .message { padding:12px; margin-top:12px; background:var(--secondary-background-color); border-radius:8px; }
        .samples { display:flex; gap:6px; flex-wrap:wrap; margin:12px 0; }
        .sample { padding:6px 9px; border-radius:999px; background:var(--secondary-background-color); }
        .sample.done { outline:2px solid var(--primary-color); }
        .warning { color:var(--warning-color, #e67e22); }
      </style>
      <div class="wrap">
        <h1>Speaker Recognition</h1>
        ${!s ? `<div class="card">Loading…</div>` : !s.configured ? `<div class="card"><strong>Finish setting up the Speaker Recognition integration first.</strong></div>` : `
          <div class="card">
            <h2>Enroll or retrain a voice</h2>
            <label>User</label>
            <select id="userSelect">${s.users.map(u => `<option value="${u.id}" ${u.id === this._userId ? "selected" : ""}>${this._escape(u.name)}</option>`).join("")}</select>
            <div class="samples">${s.phrases.map((_, i) => `<button class="sample ${staged.includes(i) ? "done" : "secondary"}" data-sample="${i}">${i + 1}${staged.includes(i) ? " ✓" : ""}</button>`).join("")}</div>
            <div class="phrase"><strong>Phrase ${this._sampleIndex + 1}:</strong><br>${this._escape(phrase)}</div>
            <h3>Record with this device</h3>
            ${micAvailable ? `<div class="row">
              ${this._recording ? `<button id="stopBtn">Stop recording</button>` : `<button id="recordBtn" ${this._busy ? "disabled" : ""}>Start recording</button>`}
              ${this._lastWav ? `<button id="playBtn" class="secondary">Play back</button><button id="useBtn" ${this._busy ? "disabled" : ""}>Use this recording</button><button id="testBtn" class="secondary" ${this._busy ? "disabled" : ""}>Test profile</button>` : ""}
            </div>` : `<p class="warning">Microphone access is unavailable in this browser context. Browser microphone APIs require HTTPS (or localhost). You can still use a compatible voice satellite or the existing WAV upload flow in the integration options.</p>`}
            <h3>Record with a voice satellite</h3>
            ${s.satellites.length ? `<div class="row"><select id="satelliteSelect">${s.satellites.map(x => `<option value="${x.entity_id}" ${x.entity_id === this._satelliteId ? "selected" : ""}>${this._escape(x.name || x.entity_id)}${x.available ? "" : " (unavailable)"}</option>`).join("")}</select><button id="satelliteBtn" ${this._busy ? "disabled" : ""}>Prompt satellite</button></div>` : `<p class="muted">No Assist Satellite entity currently advertises remote Start Conversation support.</p>`}
            <p class="muted">The satellite path is bound to the selected satellite and the exact Assist turn; unrelated speech from another satellite is ignored.</p>
            <div class="row"><button id="commitBtn" ${staged.length < minimum || this._busy ? "disabled" : ""}>Train with ${staged.length} staged sample${staged.length === 1 ? "" : "s"}</button><span class="muted">Minimum ${minimum}; up to ${s.phrases.length}.</span></div>
            ${this._message ? `<div class="message">${this._escape(this._message)}</div>` : ""}
          </div>
          <div class="card"><h2>Profile diagnostics</h2><p>Enrolled users: ${s.enrolled_users.length ? s.enrolled_users.map(x => this._escape(x)).join(", ") : "none"}</p><p class="muted">Record any phrase above and choose <strong>Test profile</strong> to see the candidate, similarity, runner-up margin and accepted/unknown decision.</p></div>
        `}
      </div>`;
    this._bindEvents();
  }

  _escape(value) {
    return String(value ?? "").replace(/[&<>'"]/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;" })[ch]);
  }

  _bindEvents() {
    const $ = (id) => this.shadowRoot.getElementById(id);
    if ($("userSelect")) $("userSelect").onchange = (e) => { this._userId = e.target.value; this._sampleIndex = 0; this._render(); };
    if ($("satelliteSelect")) $("satelliteSelect").onchange = (e) => { this._satelliteId = e.target.value; };
    if ($("recordBtn")) $("recordBtn").onclick = () => this._startRecording();
    if ($("stopBtn")) $("stopBtn").onclick = () => this._stopRecording();
    if ($("playBtn")) $("playBtn").onclick = () => this._playRecording();
    if ($("useBtn")) $("useBtn").onclick = () => this._useRecording();
    if ($("testBtn")) $("testBtn").onclick = () => this._testProfile();
    if ($("satelliteBtn")) $("satelliteBtn").onclick = () => this._startSatellite();
    if ($("commitBtn")) $("commitBtn").onclick = () => this._commitEnrollment();
    for (const button of this.shadowRoot.querySelectorAll("[data-sample]")) {
      button.onclick = () => { this._sampleIndex = Number(button.dataset.sample); this._lastWav = null; this._render(); };
    }
  }
}

customElements.define("speaker-recognition-panel", SpeakerRecognitionPanel);
