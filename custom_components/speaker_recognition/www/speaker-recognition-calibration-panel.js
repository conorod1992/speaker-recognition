import "./speaker-recognition-panel.js";

const BasePanel = customElements.get("speaker-recognition-panel");

class SpeakerRecognitionCalibrationPanel extends BasePanel {
  constructor() {
    super();
    this._calibration = null;
    this._calibrationEntryId = "";
    this._calibrationMessage = "";
    this._calibrationBusy = false;
  }

  async _refreshHistory(silent = false) {
    await super._refreshHistory(silent);
    await this._refreshCalibration(true);
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

  _renderCalibrationCard() {
    const entries = this._calibration && this._calibration.conversation_entries
      ? this._calibration.conversation_entries
      : [];
    if (!entries.length) {
      return `<div class="card"><h2>Threshold guidance</h2><p class="muted">Add a Speaker Recognition Conversation proxy to calibrate its Home Assistant confidence threshold.</p></div>`;
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
        <p class="muted">Keep marking real Assist results as Correct, Wrong speaker, or Should have recognised me. No threshold recommendation is made until there is enough evidence.</p>
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
      <p class="muted">Uses only the explicit feedback you provide on normal Assist decisions. It simulates the Home Assistant confidence threshold and deliberately treats a wrong-person identity as much more costly than a missed recognition.</p>
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

  _render() {
    super._render();
    if (!this.shadowRoot || !this._status || !this._status.configured) return;
    const wrap = this.shadowRoot.querySelector(".wrap");
    if (!wrap) return;
    const holder = document.createElement("div");
    holder.innerHTML = this._renderCalibrationCard();
    const card = holder.firstElementChild;
    if (card) wrap.appendChild(card);
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
