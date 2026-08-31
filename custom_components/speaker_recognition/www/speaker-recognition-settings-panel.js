import "./speaker-recognition-calibration-panel.js";

const BasePanel = customElements.get("speaker-recognition-calibration-panel");

class SpeakerRecognitionSettingsPanel extends BasePanel {
  constructor() {
    super();
    this._settings = null;
    this._settingsMessage = "";
    this._settingsBusy = "";
    this._panelSection = "enrollment";
  }

  _pollForSatelliteCapture(index, sessionId) {
    if (this._pollTimer) clearTimeout(this._pollTimer);
    const started = Date.now();
    const poll = async () => {
      if (!this.isConnected) return;
      await this._refresh(true);
      if (!this.isConnected || !this._status) return;
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
      if (this.isConnected) this._pollTimer = setTimeout(poll, 1000);
    };
    if (this.isConnected) this._pollTimer = setTimeout(poll, 1000);
  }

  _pollForLiveTest(sessionId) {
    if (this._livePollTimer) clearTimeout(this._livePollTimer);
    const started = Date.now();
    const poll = async () => {
      if (!this.isConnected) return;
      await this._refresh(true);
      if (!this.isConnected || !this._status) return;
      const result = this._status.live_test_result;
      if (result && result.session_id === sessionId) {
        this._liveBusy = false;
        this._liveMessage = "Live test completed.";
        await this._refreshHistory(true);
        if (!this.isConnected) return;
        this._render();
        return;
      }
      if (Date.now() - started > 90000) {
        this._liveBusy = false;
        this._liveMessage = "No matching Assist turn was seen within 90 seconds. Make sure this pipeline uses the Speaker Recognition STT and Conversation proxies.";
        this._render();
        return;
      }
      if (this.isConnected) this._livePollTimer = setTimeout(poll, 1000);
    };
    if (this.isConnected) this._livePollTimer = setTimeout(poll, 1000);
  }

  async _refreshHistory(silent = false) {
    await super._refreshHistory(silent);
    await this._refreshSettings(true);
  }

  async _refreshSettings(silent = false) {
    if (!this._hass) return;
    try {
      this._settings = await this._call({ type: "speaker_recognition/settings" });
      if (!silent) this._settingsMessage = "";
    } catch (err) {
      this._settingsMessage = this._errorText(err);
    }
    this._render();
  }

  _userName(userId) {
    if (!userId || !this._status) return userId || "Unknown";
    const user = this._status.users.find(item => item.id === userId);
    if (user) return user.name;
    const value = String(userId);
    if (value.length <= 12) return `Unknown HA user (${value})`;
    return `Unknown HA user (${value.slice(0, 6)}…${value.slice(-4)})`;
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
        const candidate = result.candidate_user_id
          ? this._userName(result.candidate_user_id)
          : "Unknown";
        this._message = `Candidate: ${candidate} · similarity ${similarity} · margin ${margin} · ${result.accepted ? "accepted" : "unknown/rejected"}`;
      }
    } catch (err) {
      this._message = this._errorText(err);
    } finally {
      this._busy = false;
      this._render();
    }
  }

  _entityOptions(domain, current) {
    const ids = this._hass
      ? Object.keys(this._hass.states).filter(entityId => entityId.startsWith(`${domain}.`))
      : [];
    if (current && !ids.includes(current)) ids.push(current);
    ids.sort();
    return ids.map(entityId => {
      const state = this._hass && this._hass.states[entityId];
      const friendly = state && state.attributes && state.attributes.friendly_name
        ? state.attributes.friendly_name
        : entityId;
      const label = friendly === entityId ? entityId : `${friendly} (${entityId})`;
      return `<option value="${this._escape(entityId)}" ${entityId === current ? "selected" : ""}>${this._escape(label)}</option>`;
    }).join("");
  }

  _renderSettingsCard() {
    const settings = this._settings;
    if (!settings) {
      return `<div class="card" id="settingsCard"><h2>Settings</h2><p class="muted">Loading settings…</p></div>`;
    }

    const main = settings.main;
    const sttEntries = settings.stt_entries || [];
    const conversationEntries = settings.conversation_entries || [];
    const fieldStyle = "width:100%;max-width:620px;padding:9px;border-radius:6px;background:var(--card-background-color);color:var(--primary-text-color);border:1px solid var(--divider-color);font:inherit";

    const backend = main ? `<div class="result">
      <strong>Recognition backend</strong>
      <label for="backendUrl">Backend URL</label>
      <input id="backendUrl" style="${fieldStyle}" value="${this._escape(main.backend_url || "")}">
      <div class="row" style="margin-top:12px"><button id="saveMainSettings" ${this._settingsBusy ? "disabled" : ""}>Save backend</button></div>
    </div>` : "";

    const stt = sttEntries.length ? sttEntries.map((entry, index) => `<div class="result" data-settings-entry="${this._escape(entry.entry_id)}">
      <strong>${this._escape(entry.title || `STT proxy ${index + 1}`)}</strong>
      <label for="sttEntity-${index}">Speech-to-text provider</label>
      <select id="sttEntity-${index}" data-stt-entity="${this._escape(entry.entry_id)}">${this._entityOptions("stt", entry.stt_entity)}</select>
      <label style="display:flex;gap:10px;align-items:center;font-weight:600;margin-top:14px">
        <input type="checkbox" data-dsp="${this._escape(entry.entry_id)}" ${entry.use_basic_dsp ? "checked" : ""}>
        Use basic DSP for speech-to-text
      </label>
      <p class="muted">Filters the audio sent to the wrapped STT provider. Speaker recognition and whisper detection continue to use the original audio.</p>
      <button data-save-stt="${this._escape(entry.entry_id)}" ${this._settingsBusy ? "disabled" : ""}>Save STT settings</button>
    </div>`).join("") : `<p class="muted">No Speaker Recognition STT proxy is configured.</p>`;

    const conversation = conversationEntries.length ? conversationEntries.map((entry, index) => `<div class="result" data-settings-entry="${this._escape(entry.entry_id)}">
      <strong>${this._escape(entry.title || `Conversation proxy ${index + 1}`)}</strong>
      <label for="conversationEntity-${index}">Conversation agent</label>
      <select id="conversationEntity-${index}" data-conversation-entity="${this._escape(entry.entry_id)}">${this._entityOptions("conversation", entry.conversation_entity)}</select>
      <label for="confidence-${index}">Minimum identity confidence: <span data-confidence-label="${this._escape(entry.entry_id)}">${Number(entry.min_confidence || 0).toFixed(2)}</span></label>
      <input id="confidence-${index}" type="range" min="0" max="1" step="0.05" value="${Number(entry.min_confidence || 0)}" data-confidence="${this._escape(entry.entry_id)}" style="width:100%;max-width:620px">
      <p class="muted">A recognised speaker is only applied to the Conversation proxy when the backend score meets this threshold.</p>
      <button data-save-conversation="${this._escape(entry.entry_id)}" ${this._settingsBusy ? "disabled" : ""}>Save Conversation settings</button>
    </div>`).join("") : `<p class="muted">No Speaker Recognition Conversation proxy is configured.</p>`;

    return `<div class="card" id="settingsCard">
      <h2>Settings</h2>
      <p class="muted">These are the live settings used by your configured Speaker Recognition entries. Saving an entry reloads that proxy so the new value takes effect.</p>
      ${this._settingsMessage ? `<div class="message">${this._escape(this._settingsMessage)}</div>` : ""}
      ${backend}
      <h3>Speech-to-text</h3>
      ${stt}
      <h3>Conversation</h3>
      ${conversation}
    </div>`;
  }

  async _saveSettings(message, busyKey) {
    this._settingsBusy = busyKey;
    this._settingsMessage = "Saving settings…";
    this._render();
    try {
      await this._call({ type: "speaker_recognition/update_settings", ...message });
      this._settingsMessage = "Settings saved.";
      await this._refreshSettings(true);
    } catch (err) {
      this._settingsMessage = this._errorText(err);
    } finally {
      this._settingsBusy = "";
      this._render();
    }
  }

  _cardByHeading(wrap, heading) {
    return [...wrap.querySelectorAll(":scope > .card")].find(card => {
      const title = card.querySelector("h2");
      return title && title.textContent.trim() === heading;
    }) || null;
  }

  _panelSectionForCard(card) {
    const heading = card.querySelector("h2");
    const title = heading ? heading.textContent.trim() : "";
    if (title === "Enroll or retrain a voice") return "enrollment";
    if (title === "Profile diagnostics" || title === "Live satellite test") return "diagnostics";
    if (title === "Recognition calibration" || title === "Threshold guidance") return "calibration";
    if (title === "Settings") return "settings";
    return "diagnostics";
  }

  _installUxStyles() {
    if (!this.shadowRoot || this.shadowRoot.getElementById("panel-ux-style")) return;
    const style = document.createElement("style");
    style.id = "panel-ux-style";
    style.textContent = `
      .panelTabs {
        display:flex;
        gap:4px;
        margin:8px 0 18px;
        padding:4px;
        overflow-x:auto;
        border-radius:12px;
        background:var(--secondary-background-color);
      }
      .panelTab {
        flex:1 0 auto;
        min-width:120px;
        padding:10px 14px;
        border-radius:9px;
        background:transparent;
        color:var(--primary-text-color);
        border:0;
        box-shadow:none;
        font-weight:600;
      }
      .panelTab.active {
        background:var(--card-background-color);
        color:var(--primary-color);
        box-shadow:var(--ha-card-box-shadow, 0 1px 4px rgba(0,0,0,.14));
      }
      .card[data-panel-section][hidden] { display:none !important; }
      .enrollment-card h2 { margin-bottom:16px; }
      .profileSummary {
        margin:14px 0 18px;
        padding:12px 14px;
        border:1px solid var(--divider-color);
        border-radius:10px;
        background:var(--secondary-background-color);
      }
      .profileSummaryChips { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
      .profileChip {
        display:inline-flex;
        align-items:center;
        min-height:28px;
        padding:4px 9px;
        border-radius:999px;
        background:var(--card-background-color);
        border:1px solid var(--divider-color);
        font-size:.9rem;
        font-weight:600;
      }
      .profileChip.ready { border-color:var(--success-color, #43a047); }
      .profileSummary p { margin:9px 0 0; }
      .sampleProgress { display:flex; justify-content:space-between; gap:12px; margin:18px 0 5px; align-items:baseline; }
      .sampleProgress span { color:var(--secondary-text-color); }
      .samples { margin-top:8px !important; }
      .sample { min-width:34px; transition:background .12s ease, outline-color .12s ease; }
      .sample.active { outline:2px solid var(--primary-color); outline-offset:2px; }
      .sample.done { background:var(--card-background-color); }
      .sample.done:not(.active) { outline:1px solid var(--primary-color); }
      .phrase { margin-top:14px; }
      .phrase strong { display:block; margin-bottom:4px; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; color:var(--secondary-text-color); }
      .enrollment-card h3 { margin:24px 0 10px; padding-top:18px; border-top:1px solid var(--divider-color); }
      .enrollment-card select { width:min(100%, 420px); }
      .trainingAction { margin-top:22px; padding-top:18px; border-top:1px solid var(--divider-color); }
      .message { border-left:4px solid var(--primary-color); }
      .message.message-success { border-left-color:var(--success-color, #43a047); }
      .message.message-error { border-left-color:var(--error-color, #db4437); }
      .profileNames { display:flex; gap:8px; flex-wrap:wrap; margin-top:8px; }
      .profileName { display:inline-block; padding:5px 9px; border-radius:999px; background:var(--secondary-background-color); border:1px solid var(--divider-color); }
      #settingsCard .result { border:1px solid var(--divider-color); background:transparent; }
      @media (max-width: 700px) {
        :host { padding:12px !important; }
        .wrap { max-width:none !important; }
        .card { padding:16px !important; }
        .panelTabs { margin-left:-2px; margin-right:-2px; }
        .panelTab { min-width:105px; }
        select { width:100%; min-width:0 !important; max-width:100% !important; }
        .row { align-items:stretch !important; }
        .row > button { flex:0 0 auto; }
        .sampleProgress { align-items:flex-start; flex-direction:column; gap:2px; }
        .metrics { grid-template-columns:1fr !important; }
      }
    `;
    this.shadowRoot.append(style);
  }

  _enhanceEnrollmentCard(wrap) {
    const card = this._cardByHeading(wrap, "Enroll or retrain a voice");
    if (!card || !this._status) return;
    card.classList.add("enrollment-card");
    const heading = card.querySelector("h2");
    if (heading) heading.textContent = "Voice enrollment";

    const staged = this._stagedIndexes();
    const minimum = Number(this._status.minimum_samples || 5);
    const total = (this._status.phrases || []).length;
    const enrolled = (this._status.enrolled_users || []).includes(this._userId);
    const remaining = Math.max(0, minimum - staged.length);

    const existingStatus = card.querySelector("#selectedEnrollmentStatus");
    if (existingStatus) {
      existingStatus.className = "profileSummary";
      existingStatus.innerHTML = `
        <div class="profileSummaryChips">
          <span class="profileChip ${enrolled ? "ready" : ""}">${enrolled ? "✓ Enrolled" : "Not enrolled"}</span>
          <span class="profileChip">${enrolled ? "Retraining" : "New enrollment"}</span>
          <span class="profileChip ${staged.length >= minimum ? "ready" : ""}">${staged.length} of ${minimum} required samples</span>
        </div>
        <p class="muted">${enrolled
          ? "Your current trained profile remains active until the replacement is successfully trained."
          : (staged.length >= minimum ? "Enough samples are staged to train this voice profile." : "Record the phrases below to create this voice profile.")}</p>`;
    }

    const samples = card.querySelector(".samples");
    if (samples) {
      const progress = document.createElement("div");
      progress.className = "sampleProgress";
      progress.innerHTML = `<strong>Training samples</strong><span>${staged.length} of ${minimum} required · ${total} available</span>`;
      samples.insertAdjacentElement("beforebegin", progress);
      for (const button of samples.querySelectorAll("[data-sample]")) {
        const index = Number(button.dataset.sample);
        button.classList.toggle("active", index === this._sampleIndex);
        button.setAttribute("aria-current", index === this._sampleIndex ? "step" : "false");
        button.title = staged.includes(index)
          ? `Phrase ${index + 1}: sample staged`
          : `Phrase ${index + 1}: not recorded`;
      }
    }

    const phrase = card.querySelector(".phrase");
    if (phrase) {
      const label = phrase.querySelector("strong");
      if (label) label.textContent = `Phrase ${this._sampleIndex + 1} of ${total}`;
    }

    const commit = card.querySelector("#commitBtn");
    if (commit) {
      const actionRow = commit.closest(".row");
      if (actionRow) actionRow.classList.add("trainingAction");
      commit.textContent = enrolled ? "Retrain profile" : "Train profile";
      const guidance = commit.nextElementSibling;
      if (guidance) {
        guidance.textContent = remaining
          ? `${remaining} more sample${remaining === 1 ? "" : "s"} required.`
          : `${staged.length} sample${staged.length === 1 ? "" : "s"} ready to train.`;
      }
    }
  }

  _enhanceDiagnosticsCard(wrap) {
    const card = this._cardByHeading(wrap, "Profile diagnostics");
    if (!card || !this._status) return;
    const heading = card.querySelector("h2");
    if (heading) heading.textContent = "Profiles";
    const paragraphs = card.querySelectorAll("p");
    if (paragraphs.length) {
      const enrolled = this._status.enrolled_users || [];
      const names = enrolled.map(userId => this._userName(userId));
      paragraphs[0].innerHTML = `<strong>Enrolled voices</strong>${names.length
        ? `<div class="profileNames">${names.map(name => `<span class="profileName">${this._escape(name)}</span>`).join("")}</div>`
        : `<span class="muted"> None</span>`}`;
    }
    if (paragraphs.length > 1) {
      paragraphs[1].innerHTML = "To test a profile directly, record any phrase in <strong>Enrollment</strong> and choose <strong>Test profile</strong>. Use the live satellite test below for a real Assist-path check.";
    }
  }

  _classifyMessages() {
    if (!this.shadowRoot) return;
    for (const message of this.shadowRoot.querySelectorAll(".message")) {
      const text = message.textContent.toLowerCase();
      message.classList.remove("message-success", "message-error");
      if (/saved|captured|completed|committed|updated|success/.test(text)) {
        message.classList.add("message-success");
      } else if (/failed|could not|no matching|unavailable|error|not found/.test(text)) {
        message.classList.add("message-error");
      }
    }
  }

  _organizePanel() {
    if (!this.shadowRoot || !this._status || !this._status.configured) return;
    const wrap = this.shadowRoot.querySelector(".wrap");
    if (!wrap) return;

    this._installUxStyles();
    this._enhanceEnrollmentCard(wrap);
    this._enhanceDiagnosticsCard(wrap);

    const tabs = document.createElement("nav");
    tabs.className = "panelTabs";
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Speaker Recognition sections");
    const sections = [
      ["enrollment", "Enrollment"],
      ["diagnostics", "Diagnostics"],
      ["calibration", "Calibration"],
      ["settings", "Settings"],
    ];
    tabs.innerHTML = sections.map(([key, label]) => `<button class="panelTab ${key === this._panelSection ? "active" : ""}" data-panel-tab="${key}" role="tab" aria-selected="${key === this._panelSection ? "true" : "false"}">${label}</button>`).join("");
    const title = wrap.querySelector("h1");
    if (title) title.insertAdjacentElement("afterend", tabs);

    for (const card of wrap.querySelectorAll(":scope > .card")) {
      const section = this._panelSectionForCard(card);
      card.dataset.panelSection = section;
      card.hidden = section !== this._panelSection;
    }
    this._classifyMessages();
  }

  _bindPanelTabs() {
    if (!this.shadowRoot) return;
    for (const button of this.shadowRoot.querySelectorAll("[data-panel-tab]")) {
      button.onclick = () => {
        this._panelSection = button.dataset.panelTab;
        this._render();
      };
    }
  }

  _render() {
    super._render();
    if (!this.shadowRoot || !this._status || !this._status.configured) return;
    const wrap = this.shadowRoot.querySelector(".wrap");
    if (!wrap) return;
    const holder = document.createElement("div");
    holder.innerHTML = this._renderSettingsCard();
    const card = holder.firstElementChild;
    if (card) wrap.appendChild(card);
    this._organizePanel();
    this._bindSettingsEvents();
    this._bindPanelTabs();
  }

  _bindSettingsEvents() {
    if (!this.shadowRoot || !this._settings) return;
    const main = this._settings.main;
    const saveMain = this.shadowRoot.getElementById("saveMainSettings");
    if (main && saveMain) {
      saveMain.onclick = () => {
        const input = this.shadowRoot.getElementById("backendUrl");
        this._saveSettings({ entry_id: main.entry_id, backend_url: input.value }, main.entry_id);
      };
    }

    for (const entry of this._settings.stt_entries || []) {
      const save = this.shadowRoot.querySelector(`[data-save-stt="${CSS.escape(entry.entry_id)}"]`);
      if (!save) continue;
      save.onclick = () => {
        const entity = this.shadowRoot.querySelector(`[data-stt-entity="${CSS.escape(entry.entry_id)}"]`);
        const dsp = this.shadowRoot.querySelector(`[data-dsp="${CSS.escape(entry.entry_id)}"]`);
        this._saveSettings(
          { entry_id: entry.entry_id, stt_entity: entity.value, use_basic_dsp: dsp.checked },
          entry.entry_id,
        );
      };
    }

    for (const entry of this._settings.conversation_entries || []) {
      const slider = this.shadowRoot.querySelector(`[data-confidence="${CSS.escape(entry.entry_id)}"]`);
      const label = this.shadowRoot.querySelector(`[data-confidence-label="${CSS.escape(entry.entry_id)}"]`);
      if (slider && label) slider.oninput = () => { label.textContent = Number(slider.value).toFixed(2); };
      const save = this.shadowRoot.querySelector(`[data-save-conversation="${CSS.escape(entry.entry_id)}"]`);
      if (!save) continue;
      save.onclick = () => {
        const entity = this.shadowRoot.querySelector(`[data-conversation-entity="${CSS.escape(entry.entry_id)}"]`);
        this._saveSettings(
          { entry_id: entry.entry_id, conversation_entity: entity.value, min_confidence: Number(slider.value) },
          entry.entry_id,
        );
      };
    }
  }
}

customElements.define("speaker-recognition-settings-panel", SpeakerRecognitionSettingsPanel);
