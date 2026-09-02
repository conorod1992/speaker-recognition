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
    if (!decisions.length) return `<p class="muted">No recent normal Assist decisions are waiting for review yet.</p>`;
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
          : `<button class="secondary reviewPlay" data-review-play="${this._escape(item.decision_id)}">▶ Play clip</button>`)
        : `<span class="muted">Audio unavailable</span>`;

      let feedback = "";
      if (item.feedback) {
        const labels = {
          correct: "Marked correct",
          wrong_speaker: "Marked wrong person",
          missed_speaker: "Marked missed speaker",
        };
        const actual = item.actual_user_id
          ? ` · ${this._escape(this._userName(item.actual_user_id))}`
          : (item.feedback === "wrong_speaker" ? " · someone not enrolled" : "");
        feedback = `<span class="feedback-saved">${labels[item.feedback] || this._escape(item.feedback)}${actual}</span>`;
      } else if (soleUser) {
        feedback = applied
          ? `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correct</button>
              <button class="secondary" data-review-feedback="wrong_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__unknown__">Not me</button>
            </div>`
          : `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correctly unknown</button>
              <button class="secondary" data-review-feedback="missed_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="${this._escape(soleUser)}">That was me</button>
            </div>`;
      } else {
        feedback = applied
          ? `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correct</button>
              <button class="secondary" data-review-feedback="wrong_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__selected__">Wrong person</button>
            </div>`
          : `<div class="feedback-actions compactFeedback">
              <button data-review-feedback="correct" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="">Correctly unknown</button>
              <button class="secondary" data-review-feedback="missed_speaker" data-review-decision="${this._escape(item.decision_id)}" data-review-actual="__selected__">Should recognise speaker</button>
            </div>`;
      }

      return `<div class="decision reviewDecision">
        <div class="reviewDecisionTop"><div><strong>${outcome}</strong>${when ? `<span class="decisionTime">${this._escape(when)}</span>` : ""}</div>${audio}</div>
        ${feedback}
        <details class="decisionDiagnostics">
          <summary>Diagnostics</summary>
          <div class="muted">Candidate ${candidate} · similarity ${Number(item.similarity || 0).toFixed(3)} · margin ${margin}</div>
          <div class="muted">Recognition ${this._formatMs(item.recognition_seconds)} · added Assist latency ${this._formatMs(item.added_latency_seconds)} · STT ${this._formatMs(item.stt_seconds)}${item.audio_seconds == null ? "" : ` · audio ${Number(item.audio_seconds).toFixed(1)} s`}</div>
        </details>
      </div>`;
    }).join("");
  }

  _bindEvents() {
    super._bindEvents();
    if (!this.shadowRoot) return;
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
  }

  _renderEnrollmentStatus() {
    if (!this._status || !this._userId) return "";
    const enrolled = (this._status.enrolled_users || []).includes(this._userId);
    const staged = this._stagedIndexes();
    const minimum = Number(this._status.minimum_samples || 5);
    const total = (this._status.phrases || []).length;
    const remaining = Math.max(0, minimum - staged.length);
    const ready = staged.length >= minimum;
    const mode = enrolled ? "Retraining existing profile" : "New enrollment";
    let guidance;
    if (ready) {
      guidance = `Ready to train. ${staged.length} new sample${staged.length === 1 ? " is" : "s are"} staged.`;
    } else if (staged.length) {
      guidance = `${remaining} more sample${remaining === 1 ? "" : "s"} needed before training.`;
    } else {
      guidance = enrolled
        ? "No replacement samples staged yet. The current trained profile remains active."
        : "No samples staged yet.";
    }
    return `<div class="result" id="selectedEnrollmentStatus">
      <strong>Selected user status</strong>
      <div class="metrics">
        <span><b>Current profile</b><br>${enrolled ? "Enrolled" : "Not enrolled"}</span>
        <span><b>Enrollment mode</b><br>${mode}</span>
        <span><b>New samples</b><br>${staged.length} staged<br><small>${minimum} minimum · ${total} available</small></span>
        <span><b>Ready to train</b><br>${ready ? "Yes" : "No"}</span>
      </div>
      <p class="muted">${guidance}${enrolled && staged.length ? " Your existing trained profile stays in use until the replacement is successfully committed." : ""}</p>
    </div>`;
  }

  _renderCalibrationCard() {
    const entries = this._calibration && this._calibration.conversation_entries
      ? this._calibration.conversation_entries
      : [];
    const decisions = this._history && this._history.decisions ? this._history.decisions : [];
    const labelled = decisions.filter(item => item.feedback).length;
    if (!entries.length) {
      return `<div class="card" id="calibrationGuidanceCard">
        <h2>Threshold guidance</h2>
        <p><strong>${decisions.length} recent recognition decision${decisions.length === 1 ? "" : "s"} available</strong>${labelled ? ` · ${labelled} labelled` : ""}</p>
        <p class="muted">The review queue above keeps only ten recent clips. Compact labelled decision metadata can continue contributing to calibration after its audio has expired. Add a Speaker Recognition Conversation proxy only if you want this section to recommend and apply a Home Assistant identity-confidence threshold.</p>
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
        <strong>More labelled decisions needed</strong>
        <p>${analysis.labelled_count} of ${analysis.minimum_labelled} labelled decisions collected.</p>
        <p class="muted">Keep reviewing real Assist results. Older reviewed audio can expire from the ten-item queue while its compact feedback remains useful for threshold calibration.</p>
      </div>`;
    } else {
      const current = Number(analysis.current_threshold).toFixed(2);
      const recommended = Number(analysis.recommended_threshold).toFixed(2);
      const unchanged = current === recommended;
      guidance = `<div class="result ${unchanged ? "success" : ""}">
        <strong>${unchanged ? "Current threshold already fits the labelled evidence" : `Suggested threshold: ${recommended}`}</strong>
        <div class="metrics">
          <span><b>Current</b><br>${current}<br><small>${this._escape(this._metricsText(analysis.current_metrics))}</small></span>
          <span><b>Suggested</b><br>${recommended}<br><small>${this._escape(this._metricsText(analysis.recommended_metrics))}</small></span>
          <span><b>Evidence</b><br>${analysis.labelled_count} labelled turns</span>
          <span><b>Error weighting</b><br>wrong person ×${analysis.false_accept_weight}<br>missed ×${analysis.missed_speaker_weight}</span>
        </div>
        ${analysis.backend_rejected_misses ? `<p class="muted">${analysis.backend_rejected_misses} labelled missed recognition${analysis.backend_rejected_misses === 1 ? " was" : "s were"} already rejected by the backend. Changing the HA threshold cannot fix ${analysis.backend_rejected_misses === 1 ? "that case" : "those cases"}; profile/enrollment quality or backend decision settings are the relevant layer.</p>` : ""}
        ${analysis.threshold_actionable_misses ? `<p class="muted">${analysis.threshold_actionable_misses} missed recognition${analysis.threshold_actionable_misses === 1 ? " appears" : "s appear"} potentially recoverable by the HA threshold.</p>` : ""}
        <button id="applyCalibrationBtn" ${unchanged || this._calibrationBusy ? "disabled" : ""}>${this._calibrationBusy ? "Applying…" : `Apply suggested threshold ${recommended}`}</button>
      </div>`;
    }

    return `<div class="card" id="calibrationGuidanceCard">
      <h2>Threshold guidance</h2>
      <p class="muted">Uses the explicit feedback you provide on normal Assist decisions. It simulates the Home Assistant confidence threshold and treats a wrong-person identity as much more costly than a missed recognition.</p>
      ${entries.length > 1 ? `<label for="calibrationEntrySelect">Conversation proxy</label><select id="calibrationEntrySelect">${options}</select>` : `<p><strong>Conversation proxy:</strong> ${this._escape(entry.title || entry.conversation_entity || entry.entry_id)}</p>`}
      ${this._calibrationMessage ? `<div class="message">${this._escape(this._calibrationMessage)}</div>` : ""}
      ${guidance}
    </div>`;
  }

  async _applyCalibration() {
    const entry = this._selectedCalibrationEntry();
    if (!entry) return;
    this._calibrationBusy = true;
    this._calibrationMessage = "Applying the current server-side recommendation…";
    this._render();
    try {
      const result = await this._call({
        type: "speaker_recognition/apply_recommended_threshold",
        entry_id: entry.entry_id,
      });
      this._calibrationMessage = `Threshold updated from ${Number(result.previous_threshold).toFixed(2)} to ${Number(result.new_threshold).toFixed(2)}.`;
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
      const intro = reviewCard.querySelector("h2 + p.muted");
      if (intro) intro.textContent = "Review the newest ten Assist decisions. Recent clips are playable; when an eleventh decision arrives the oldest clip is discarded automatically.";
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
    const apply = this.shadowRoot.getElementById("applyCalibrationBtn");
    if (apply) apply.onclick = () => this._applyCalibration();
  }
}

customElements.define("speaker-recognition-calibration-panel", SpeakerRecognitionCalibrationPanel);
import("./speaker-recognition-enhancement-panel.js");
