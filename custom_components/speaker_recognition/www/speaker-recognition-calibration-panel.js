import "./speaker-recognition-panel.js";

const BasePanel = customElements.get("speaker-recognition-panel");

class SpeakerRecognitionCalibrationPanel extends BasePanel {
  constructor() {
    super();
    this._calibration = null;
    this._calibrationEntryId = "";
    this._calibrationMessage = "";
    this._calibrationBusy = false;
    this._reviewAudioUrls = new Map();
  }

  disconnectedCallback() {
    for (const url of this._reviewAudioUrls.values()) URL.revokeObjectURL(url);
    this._reviewAudioUrls.clear();
    super.disconnectedCallback();
  }

  async _refreshHistory(silent = false) {
    if (!this._hass) return;
    try {
      this._history = await this._call({ type: "speaker_recognition/review_decisions" });
      if (!silent) this._historyMessage = "";
      const activeIds = new Set((this._history.decisions || []).map(item => item.decision_id));
      for (const [decisionId, url] of this._reviewAudioUrls.entries()) {
        if (!activeIds.has(decisionId)) {
          URL.revokeObjectURL(url);
          this._reviewAudioUrls.delete(decisionId);
        }
      }
    } catch (err) {
      this._historyMessage = this._errorText(err);
    }
    await this._refreshCalibration(true);
    this._render();
  }

  async _refreshCalibration(silent = false) {
    if (!this._hass) return;
    try {
      this._calibration = await this._call({ type: "speaker_recognition/calibration_analysis" });
      const entries = this._calibration.conversation_entries || [];
      if (!entries.some(item => item.entry_id === this._calibrationEntryId)) {
        this._calibrationEntryId = entries.length ? entries[0].entry_id : "";
      }
      if (!silent) this._calibrationMessage = "";
    } catch (err) {
      this._calibrationMessage = this._errorText(err);
    }
    this._render();
  }

  _selectedCalibrationEntry() {
    const entries = this._calibration && this._calibration.conversation_entries
      ? this._calibration.conversation_entries
      : [];
    return entries.find(item => item.entry_id === this._calibrationEntryId) || entries[0] || null;
  }

  _metricsText(metrics) {
    if (!metrics) return "";
    return `${metrics.false_accepts} wrong-person · ${metrics.missed_speakers} missed`;
  }

  _renderLiveResult(result) {
    const rendered = super._renderLiveResult(result);
    if (!result || !rendered) return rendered;

    const whisperLabel = result.whisper_available === false
      ? "Unavailable"
      : (result.whispering ? "Yes" : "No");
    const whisperScore = result.whisper_available === false
      ? ""
      : `<br><small>score ${Number(result.whisper_score || 0).toFixed(2)}</small>`;
    const metric = `<span><b>Whispering Detected</b><br>${whisperLabel}${whisperScore}</span>`;
    let enriched = rendered.replace('<div class="metrics">', `<div class="metrics">${metric}`);

    const diagnostics = result.whisper_diagnostics || {};
    if (result.whisper_available !== false && Object.keys(diagnostics).length) {
      const percentage = value => `${Math.round(Number(value || 0) * 100)}%`;
      const fixed = value => Number(value || 0).toFixed(2);
      const diagnosticsPanel = `<details>
        <summary>Whisper diagnostics</summary>
        <p class="muted">These component measurements are shown for tuning. The overall whisper score is diagnostic, not a calibrated probability. Strong periodic vowel frames can reduce the score when otherwise whisper-like audio still contains convincing normal voicing.</p>
        <div class="metrics">
          <span><b>Voicing evidence</b><br>${fixed(diagnostics.voicing_score)}</span>
          <span><b>Spectral evidence</b><br>${fixed(diagnostics.spectral_score)}</span>
          <span><b>Normal-voice rescue</b><br>${fixed(diagnostics.normal_voicing_rescue)}</span>
          <span><b>Periodicity</b><br>${fixed(diagnostics.periodicity)}</span>
          <span><b>Peak periodicity</b><br>${fixed(diagnostics.peak_periodicity)}</span>
          <span><b>Voiced frames</b><br>${percentage(diagnostics.voiced_fraction)}</span>
          <span><b>Strong voiced frames</b><br>${percentage(diagnostics.strong_voiced_fraction)}</span>
          <span><b>Spectral flatness</b><br>${fixed(diagnostics.spectral_flatness)}</span>
          <span><b>Spectral centroid</b><br>${Math.round(Number(diagnostics.spectral_centroid_hz || 0))} Hz</span>
          <span><b>Low-band energy</b><br>${percentage(diagnostics.low_frequency_ratio)}</span>
          <span><b>High-band energy</b><br>${percentage(diagnostics.high_frequency_ratio)}</span>
          <span><b>Zero crossing rate</b><br>${fixed(diagnostics.zero_crossing_rate)}</span>
          <span><b>Difference ratio</b><br>${fixed(diagnostics.difference_ratio)}</span>
        </div>
      </details>`;
      const closingIndex = enriched.lastIndexOf("</div>");
      if (closingIndex >= 0) {
        enriched = `${enriched.slice(0, closingIndex)}${diagnosticsPanel}${enriched.slice(closingIndex)}`;
      }
    }
    return enriched;
  }

  _reviewWhen(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    const now = new Date();
    const sameDay = date.getFullYear() === now.getFullYear()
      && date.getMonth() === now.getMonth()
      && date.getDate() === now.getDate();
    const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return sameDay ? time : `${date.toLocaleDateString()} ${time}`;
  }

  _base64ToBytes(value) {
    const binary = atob(value);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    return bytes;
  }

  _reviewWavUrl(pcmBytes, sampleRate) {
    const buffer = new ArrayBuffer(44 + pcmBytes.length);
    const view = new DataView(buffer);
    const write = (offset, text) => {
      for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
    };
    write(0, "RIFF");
    view.setUint32(4, 36 + pcmBytes.length, true);
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
    view.setUint32(40, pcmBytes.length, true);
    new Uint8Array(buffer, 44).set(pcmBytes);
    return URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  }

  async _playReviewAudio(decisionId) {
    let url = this._reviewAudioUrls.get(decisionId);
    if (!url) {
      try {
        const clip = await this._call({
          type: "speaker_recognition/decision_audio",
          decision_id: decisionId,
        });
        const sampleRate = Number(clip.sample_rate || 0);
        if (!sampleRate || !clip.pcm_base64) throw new Error("The saved clip is invalid");
        url = this._reviewWavUrl(this._base64ToBytes(clip.pcm_base64), sampleRate);
        this._reviewAudioUrls.set(decisionId, url);
        this._render();
      } catch (err) {
        this._historyMessage = this._errorText(err);
        this._render();
        return;
      }
    }
    const audio = this.shadowRoot && this.shadowRoot.querySelector(`audio[data-review-audio="${decisionId}"]`);
    if (audio) audio.play().catch(() => {});
  }

  async _submitReviewFeedback(decisionId, feedback, actualUserId) {
    let actual = actualUserId;
    if (actual === "__selected__") actual = this._feedbackUserId;
    if (actual === "__unknown__" || actual === "") actual = null;
    const message = {
      type: "speaker_recognition/review_feedback",
      decision_id: decisionId,
      feedback,
      actual_user_id: actual,
    };
    try {
      await this._call(message);
      this._historyMessage = "Feedback saved.";
      await this._refreshHistory(true);
    } catch (err) {
      this._historyMessage = this._errorText(err);
      this._render();
    }
  }

  _renderHistory() {
    const decisions = this._history && this._history.decisions ? this._history.decisions : [];
    if (!decisions.length) return `<p class="muted">No recognition results are waiting for review.</p>`;
    const enrolled = this._status && Array.isArray(this._status.enrolled_users)
      ? this._status.enrolled_users : [];
    const soleUser = enrolled.length === 1 ? enrolled[0] : null;

    return decisions.slice(0, 10).map(item => {
      const applied = Boolean(item.identity_eligible && item.user_id);
      const outcome = applied
        ? `Recognised as ${this._escape(this._userName(item.user_id))}`
        : "Not recognised";
      const when = this._reviewWhen(item.created_at);
      const candidate = this._escape(this._userName(item.candidate_user_id));
      const margin = item.margin == null ? "n/a" : Number(item.margin).toFixed(3);
      const url = this._reviewAudioUrls.get(item.decision_id);
      const audio = item.has_audio
        ? (url
          ? `<audio controls preload="metadata" data-review-audio="${this._escape(item.decision_id)}" src="${this._escape(url)}"></audio>`
          : `<button class="secondary reviewPlay" data-review-play="${this._escape(item.decision_id)}">▶ Play recording</button>`)
        : `<span class="muted">Audio unavailable</span>`;

      let actions;
      if (soleUser) {
        actions = applied
          ? `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correct</button>
              <button class="secondary" data-review-feedback="wrong_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__unknown__">Not me</button>
              <button class="secondary" data-review-ignore="${this._escape(item.decision_id)}">Ignore</button>
            </div>`
          : `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correctly unknown</button>
              <button class="secondary" data-review-feedback="missed_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="${this._escape(soleUser)}">That was me</button>
              <button class="secondary" data-review-ignore="${this._escape(item.decision_id)}">Ignore</button>
            </div>`;
      } else {
        actions = applied
          ? `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correct</button>
              <button class="secondary" data-review-feedback="wrong_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__selected__">Wrong person</button>
              <button class="secondary" data-review-ignore="${this._escape(item.decision_id)}">Ignore</button>
            </div>`
          : `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correctly unknown</button>
              <button class="secondary" data-review-feedback="missed_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__selected__">Should recognise speaker</button>
              <button class="secondary" data-review-ignore="${this._escape(item.decision_id)}">Ignore</button>
            </div>`;
      }

      return `<div class="decision reviewDecision">
        <div class="reviewDecisionTop"><div><strong>${outcome}</strong>${when ? `<span class="decisionTime">${this._escape(when)}</span>` : ""}</div>${audio}</div>
        ${actions}
        <details class="decisionDiagnostics">
          <summary>Technical details</summary>
          <div class="muted">Best match ${candidate} · similarity ${Number(item.similarity || 0).toFixed(3)} · margin ${margin}</div>
          <div class="muted">Recognition ${this._formatMs(item.recognition_seconds)} · added delay ${this._formatMs(item.added_latency_seconds)} · speech-to-text ${this._formatMs(item.stt_seconds)}${item.audio_seconds == null ? "" : ` · audio ${Number(item.audio_seconds).toFixed(1)} s`}</div>
        </details>
      </div>`;
    }).join("");
  }

  async _dismissReview(decisionId) {
    try {
      await this._call({ type: "speaker_recognition/review_dismiss", decision_id: decisionId });
      this._historyMessage = "Recognition result ignored.";
      await this._refreshHistory(true);
    } catch (err) {
      this._historyMessage = this._errorText(err);
      this._render();
    }
  }

  async _dismissAllReviews() {
    if (!window.confirm("Ignore all recognition results currently waiting for review?")) return;
    try {
      const result = await this._call({ type: "speaker_recognition/review_dismiss_all" });
      const count = Number(result.dismissed || 0);
      this._historyMessage = count
        ? `${count} recognition result${count === 1 ? "" : "s"} ignored.`
        : "No pending recognition results to ignore.";
      await this._refreshHistory(true);
    } catch (err) {
      this._historyMessage = this._errorText(err);
      this._render();
    }
  }

  _bindEvents() {
    super._bindEvents();
    if (!this.shadowRoot) return;
    for (const button of this.shadowRoot.querySelectorAll("[data-promote-decision]")) {
      button.onclick = async () => {
        button.disabled = true;
        try {
          await this._call({type: "speaker_recognition/promote_decision", decision_id: button.dataset.promoteDecision});
          this._historyMessage = "Recording added. Open Voices when you are ready to update the voice profile.";
          await this._refresh(true);
          await this._refreshHistory(true);
        } catch (err) { this._historyMessage = this._errorText(err); this._render(); }
      };
    }
    for (const button of this.shadowRoot.querySelectorAll("[data-review-play]")) {
      button.onclick = () => this._playReviewAudio(button.dataset.reviewPlay);
    }
    for (const button of this.shadowRoot.querySelectorAll("[data-review-feedback]")) {
      button.onclick = () => this._submitReviewFeedback(
        button.dataset.reviewDecision,
        button.dataset.reviewFeedback,
        button.dataset.reviewActual,
      );
    }
    for (const button of this.shadowRoot.querySelectorAll("[data-review-ignore]")) {
      button.onclick = () => this._dismissReview(button.dataset.reviewIgnore);
    }
    const ignoreAll = this.shadowRoot.getElementById("ignoreAllReviewsBtn");
    if (ignoreAll) ignoreAll.onclick = () => this._dismissAllReviews();
  }

  _renderEnrollmentStatus() {
    if (!this._status || !this._userId) return "";
    const enrolled = (this._status.enrolled_users || []).includes(this._userId);
    const staged = this._stagedIndexes();
    const minimum = Number(this._status.minimum_samples || 5);
    const total = (this._status.phrases || []).length;
    const remaining = Math.max(0, minimum - staged.length);
    const ready = staged.length >= minimum;
    let guidance;
    if (ready) {
      guidance = `Ready to ${enrolled ? "update" : "create"} this voice profile.`;
    } else if (staged.length) {
      guidance = `${remaining} more recording${remaining === 1 ? "" : "s"} needed.`;
    } else {
      guidance = enrolled
        ? `Your voice profile is active. Record at least ${minimum} of the ${total} phrases below if you want to replace it.`
        : `Record at least ${minimum} of the ${total} phrases below to create a voice profile.`;
    }
    return `<div class="result" id="selectedEnrollmentStatus">
      <strong>${enrolled ? "✓ Voice profile active" : "No voice profile yet"}</strong>
      <p>${staged.length} of ${minimum} recordings ready</p>
      <p class="muted">${guidance}${enrolled && staged.length ? " Your existing profile keeps working until the update is ready." : ""}</p>
    </div>`;
  }

  _renderBackendCalibration() {
    const backend = this._calibration?.backend;
    if (!backend) return "";
    const a = backend.analysis;
    const policy = p => `similarity ${Number(p.min_similarity).toFixed(2)} · margin ${Number(p.min_margin).toFixed(2)}`;
    const metrics = m => `${m.correct_identity} correct · ${m.wrong_speaker + m.false_accepts} wrong person · ${m.false_unknowns} missed · ${m.correct_rejection} correctly unknown`;
    const recommended = a.recommended_policy;
    const unchanged = recommended && recommended.min_similarity === a.current_policy.min_similarity && recommended.min_margin === a.current_policy.min_margin;
    if (!recommended) {
      return `<div class="result" id="backendCalibration">
        <h3>Recognition settings</h3>
        <strong>Keep reviewing recognition results</strong>
        <p>${a.labelled_count} of ${a.minimum_labelled} reviewed results collected.</p>
        <p class="muted">Once there is enough evidence, Speaker Recognition can recommend safer recognition settings.</p>
        <details><summary>Technical details</summary><p>Current: ${policy(a.current_policy)}</p><p>${this._escape(a.summary)}</p></details>
      </div>`;
    }
    return `<div class="result ${unchanged ? "success" : ""}" id="backendCalibration">
      <h3>Recognition settings</h3>
      <strong>${unchanged ? "Current recognition settings fit the reviewed results" : "A settings change may improve recognition"}</strong>
      <p>${a.labelled_count} reviewed results were used for this recommendation.</p>
      ${unchanged ? "" : `<button id="applyBackendCalibrationBtn" ${this._calibrationBusy ? "disabled" : ""}>Apply recommendation</button>`}
      <details>
        <summary>Technical details</summary>
        <p>Current: ${policy(a.current_policy)}</p>
        <p>Recommended: ${policy(recommended)}</p>
        <p>Current results: ${metrics(a.current_metrics)}</p>
        <p>Recommended results: ${metrics(a.recommended_metrics)}</p>
        <p>${this._escape(a.summary)}</p>
      </details>
    </div>`;
  }

  async _applyBackendCalibration() {
    const backend = this._calibration?.backend;
    if (!backend) return;
    this._calibrationBusy = true;
    this._render();
    try {
      await this._call({ type: "speaker_recognition/apply_recommended_backend_thresholds", entry_id: backend.entry_id });
      this._calibrationMessage = "Recognition settings updated using your reviewed results.";
      await this._refreshHistory(true);
    } catch (err) {
      this._calibrationMessage = this._errorText(err);
    } finally {
      this._calibrationBusy = false;
      this._render();
    }
  }

  _renderCalibrationCard() {
    const entries = this._calibration && this._calibration.conversation_entries
      ? this._calibration.conversation_entries
      : [];
    const decisions = this._history && this._history.decisions ? this._history.decisions : [];
    const labelled = decisions.filter(item => item.feedback).length;
    if (!entries.length) {
      return `<div class="card" id="calibrationGuidanceCard">
        <h2>Improve accuracy</h2>
        <p class="muted">Review real recognition results and Speaker Recognition can suggest settings based on how it performs in your home.</p>
        ${this._renderBackendCalibration()}
        ${this._calibrationMessage ? `<div class="message">${this._escape(this._calibrationMessage)}</div>` : ""}
        <p><strong>${labelled} recent result${labelled === 1 ? "" : "s"} reviewed</strong></p>
        <p class="muted">The latest 10 recordings can be played back. Your earlier answers can still help future recommendations.</p>
      </div>`;
    }

    const entry = this._selectedCalibrationEntry();
    const analysis = entry.analysis;
    const options = entries.map(item => {
      const label = item.title || item.conversation_entity || item.entry_id;
      return `<option value="${this._escape(item.entry_id)}" ${item.entry_id === entry.entry_id ? "selected" : ""}>${this._escape(label)}</option>`;
    }).join("");

    let guidance;
    if (!analysis.ready) {
      guidance = `<div class="result">
        <strong>Keep reviewing recognition results</strong>
        <p>${analysis.labelled_count} of ${analysis.minimum_labelled} reviewed results collected.</p>
        <p class="muted">Once there is enough evidence, Speaker Recognition can recommend whether this identity setting should change.</p>
      </div>`;
    } else {
      const current = Number(analysis.current_threshold).toFixed(2);
      const recommended = Number(analysis.recommended_threshold).toFixed(2);
      const unchanged = current === recommended;
      guidance = `<div class="result ${unchanged ? "success" : ""}">
        <strong>${unchanged ? "Current identity setting fits the reviewed results" : "A conversation identity setting may improve recognition"}</strong>
        <p>${analysis.labelled_count} reviewed results were used.</p>
        ${unchanged ? "" : `<button id="applyCalibrationBtn" ${this._calibrationBusy ? "disabled" : ""}>${this._calibrationBusy ? "Applying…" : "Apply recommendation"}</button>`}
        ${analysis.backend_rejected_misses ? `<p class="muted">Some missed recognitions happened before this identity setting was checked. Updating the affected voice profile or the advanced recognition settings may help those cases.</p>` : ""}
        <details>
          <summary>Technical details</summary>
          <p>Current identity confidence: ${current} · ${this._escape(this._metricsText(analysis.current_metrics))}</p>
          <p>Recommended identity confidence: ${recommended} · ${this._escape(this._metricsText(analysis.recommended_metrics))}</p>
          <p>Wrong-person matches are weighted more heavily than missed recognitions when choosing a recommendation.</p>
          ${analysis.threshold_actionable_misses ? `<p>${analysis.threshold_actionable_misses} missed recognition${analysis.threshold_actionable_misses === 1 ? " may be" : "s may be"} affected by this setting.</p>` : ""}
        </details>
      </div>`;
    }

    return `<div class="card" id="calibrationGuidanceCard">
      <h2>Improve accuracy</h2>
      <p class="muted">Review real recognition results and apply recommendations only when you want to.</p>
      ${this._renderBackendCalibration()}
      ${entries.length > 1 ? `<label for="calibrationEntrySelect">Conversation setup</label><select id="calibrationEntrySelect">${options}</select>` : ""}
      ${this._calibrationMessage ? `<div class="message">${this._escape(this._calibrationMessage)}</div>` : ""}
      ${guidance}
    </div>`;
  }

  async _applyCalibration() {
    const entry = this._selectedCalibrationEntry();
    if (!entry) return;
    this._calibrationBusy = true;
    this._calibrationMessage = "Applying recommendation…";
    this._render();
    try {
      await this._call({
        type: "speaker_recognition/apply_recommended_threshold",
        entry_id: entry.entry_id,
      });
      this._calibrationMessage = `Identity setting updated.`;
      await this._refreshCalibration(true);
    } catch (err) {
      this._calibrationMessage = this._errorText(err);
    } finally {
      this._calibrationBusy = false;
      this._render();
    }
  }

  _installReviewStyles() {
    if (!this.shadowRoot || this.shadowRoot.getElementById("calibration-review-style")) return;
    const style = document.createElement("style");
    style.id = "calibration-review-style";
    style.textContent = `
      .reviewDecision { padding:14px 0; }
      .reviewDecisionTop { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
      .decisionTime { margin-left:9px; color:var(--secondary-text-color); font-size:.88rem; font-weight:400; }
      .reviewDecision audio { width:min(300px, 100%); height:36px; }
      .reviewPlay { padding:7px 11px; }
      .compactFeedback { margin-top:10px; }
      .decisionDiagnostics { margin-top:10px; }
      .decisionDiagnostics summary { cursor:pointer; color:var(--secondary-text-color); }
      .decisionDiagnostics .muted { margin-top:5px; }
    `;
    this.shadowRoot.append(style);
  }

  _render() {
    super._render();
    if (!this.shadowRoot || !this._status || !this._status.configured) return;
    const wrap = this.shadowRoot.querySelector(".wrap");
    if (!wrap) return;

    const reviewCard = Array.from(wrap.querySelectorAll(".card")).find(card => {
      const heading = card.querySelector("h2");
      return heading && heading.textContent.trim() === "Recognition calibration";
    });
    if (reviewCard) {
      const heading = reviewCard.querySelector("h2");
      if (heading) heading.textContent = "Recent recognition results";
      if ((this._history?.decisions || []).length && !reviewCard.querySelector("#ignoreAllReviewsBtn")) {
        const button = document.createElement("button");
        button.id = "ignoreAllReviewsBtn";
        button.className = "secondary";
        button.textContent = "Ignore all";
        heading.insertAdjacentElement("afterend", button);
      }
      const intro = reviewCard.querySelector("h2 + p.muted");
      if (intro) intro.textContent = "Review the latest recognition results. The newest 10 recordings can be played back; older answers can still help recommendations.";
      const select = reviewCard.querySelector("#feedbackUserSelect");
      const label = reviewCard.querySelector('label[for="feedbackUserSelect"]');
      const enrolled = Array.isArray(this._status.enrolled_users) ? this._status.enrolled_users : [];
      if (select && enrolled.length === 1) {
        select.hidden = true;
        if (label) label.hidden = true;
      } else if (select && !select.querySelector('option[value="__unknown__"]')) {
        const option = document.createElement("option");
        option.value = "__unknown__";
        option.textContent = "Someone not enrolled";
        select.appendChild(option);
      }
    }

    const enrollmentCard = wrap.querySelector(".card");
    if (enrollmentCard) {
      const holder = document.createElement("div");
      holder.innerHTML = this._renderEnrollmentStatus();
      const status = holder.firstElementChild;
      const userSelect = enrollmentCard.querySelector("#userSelect");
      if (status && userSelect) {
        userSelect.insertAdjacentElement("afterend", status);
      }
    }

    const holder = document.createElement("div");
    holder.innerHTML = this._renderCalibrationCard();
    const card = holder.firstElementChild;
    if (card) wrap.appendChild(card);
    this._installReviewStyles();
    this._bindCalibrationEvents();
  }

  _bindCalibrationEvents() {
    if (!this.shadowRoot) return;
    const select = this.shadowRoot.getElementById("calibrationEntrySelect");
    if (select) {
      select.onchange = (event) => {
        this._calibrationEntryId = event.target.value;
        this._calibrationMessage = "";
        this._render();
      };
    }
    const applyBackend = this.shadowRoot.getElementById("applyBackendCalibrationBtn");
    if (applyBackend) applyBackend.onclick = () => this._applyBackendCalibration();
    const apply = this.shadowRoot.getElementById("applyCalibrationBtn");
    if (apply) apply.onclick = () => this._applyCalibration();
  }
}

customElements.define("speaker-recognition-calibration-panel", SpeakerRecognitionCalibrationPanel);
import("./speaker-recognition-enhancement-panel.js");
