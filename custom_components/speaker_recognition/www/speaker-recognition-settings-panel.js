import "./speaker-recognition-calibration-panel.js";

const BasePanel = customElements.get("speaker-recognition-calibration-panel");

class SpeakerRecognitionSettingsPanel extends BasePanel {
  constructor() {
    super();
    this._settings = null;
    this._profileHealth = null;
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
    await this._refreshProfileHealth();
  }

  async _refreshProfileHealth() {
    if (!this._hass) return;
    try {
      this._profileHealth = await this._call({ type: "speaker_recognition/profile_health" });
    } catch (_) {
      this._profileHealth = { available: false };
    }
    this._render();
  }

  _renderProfileHealth() {
    const health = this._profileHealth;
    const number = value => Number.isFinite(value) ? value.toFixed(2) : "Unavailable";
    if (!health) return '<p class="muted">Voice profile checks have not loaded yet.</p>';
    if (!health.available) return '<p class="muted">Voice profile checks are temporarily unavailable.</p>';
    if (!health.profiles.length) return '<p class="muted">No enrolled voices to check yet.</p>';

    const selected = health.profiles.find(profile => profile.user_id === this._profileHealthUserId)
      || health.profiles[0];
    if (!this._profileHealthUserId) this._profileHealthUserId = selected.user_id;

    const comparison = selected.nearest_user_id
      ? (selected.low_separation
        ? `<div class="profileHealthStatus warningStatus"><strong>Voices may be hard to tell apart</strong><p>This voice is unusually similar to ${this._escape(selected.nearest_user_name)}. Recording some new samples may improve recognition.</p></div>`
        : `<div class="profileHealthStatus successStatus"><strong>Voice looks distinct</strong><p>No unusually similar enrolled voice was found.</p></div>`)
      : `<div class="profileHealthStatus neutralStatus"><strong>No comparison needed yet</strong><p>There are no other enrolled voices to compare with this one.</p></div>`;

    return `${comparison}
      ${selected.sample_data_incomplete ? '<p class="muted">Some older profile data could not be checked.</p>' : ""}
      ${(selected.sample_warnings || []).length ? '<p><strong>Some recordings may be worth replacing.</strong></p>' : ""}
      <details class="profileTechnicalDetails">
        <summary>Technical details</summary>
        <p><strong>Internal consistency:</strong> ${number(selected.internal_consistency)}</p>
        <p class="muted">This measures how similar this voice's training recordings are to one another. Higher values usually mean the recordings are more consistent.</p>
        <p>${selected.nearest_user_id
          ? `Nearest other profile: ${this._escape(selected.nearest_user_name)} · separation: ${number(selected.separation)}`
          : "No other enrolled voices to compare."}</p>
        ${(selected.sample_warnings || []).map(sample => `<p>Stored sample ${Number(sample.sample_index)} is close to ${this._escape(sample.competing_user_name)} (gap: ${number(sample.separation)}).</p>`).join("")}
      </details>`;
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
      this._message = "Record a phrase first, then choose Test profile.";
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
        this._message = "No trained voice profile is available yet.";
      } else {
        const candidate = result.candidate_user_id
          ? this._userName(result.candidate_user_id)
          : "Unknown";
        this._message = result.accepted
          ? `Recognised as ${candidate}.`
          : "No confident voice match was found.";
      }
    } catch (err) {
      this._message = this._errorText(err);
    } finally {
      this._busy = false;
      this._render();
    }
  }

  _entityOptions(domain, current, allowed = null) {
    const ids = Array.isArray(allowed)
      ? [...allowed]
      : (this._hass
        ? Object.keys(this._hass.states).filter(entityId => entityId.startsWith(`${domain}.`))
        : []);
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

    const backend = main ? `<details class="result advancedSettings">
      <summary><strong>Advanced recognition settings</strong></summary>
      <p class="muted">These controls are mainly for troubleshooting or manual tuning. Automatic recommendations are available under Improve accuracy.</p>
      <label for="backendUrl">Recognition service address</label>
      <input id="backendUrl" style="${fieldStyle}" value="${this._escape(main.backend_url || "")}">
      <label for="backendSimilarity">Recognition strictness <span class="muted">(similarity threshold)</span></label>
      <input id="backendSimilarity" type="number" min="0" max="1" step="0.01" value="${main.acceptance_thresholds?.min_similarity ?? 0.55}">
      <p class="muted">How closely a voice must match an enrolled profile.</p>
      <label for="backendMargin">Ambiguous-match protection <span class="muted">(margin threshold)</span></label>
      <input id="backendMargin" type="number" min="0" max="1" step="0.01" value="${main.acceptance_thresholds?.min_margin ?? 0.05}">
      <p class="muted">How much better the best match must be than the next-best match.</p>
      <div class="row" style="margin-top:12px"><button id="saveMainSettings" ${this._settingsBusy ? "disabled" : ""}>Save advanced settings</button></div>
    </details>` : "";

    const stt = sttEntries.length ? sttEntries.map((entry, index) => `<div class="result" data-settings-entry="${this._escape(entry.entry_id)}">
      <strong>${this._escape(entry.title || `Speech-to-text ${index + 1}`)}</strong>
      <label for="sttEntity-${index}">Speech-to-text provider</label>
      <select id="sttEntity-${index}" data-stt-entity="${this._escape(entry.entry_id)}">${this._entityOptions("stt", entry.stt_entity, entry.stt_options)}</select>
      <label style="display:flex;gap:10px;align-items:center;font-weight:600;margin-top:14px">
        <input type="checkbox" data-dsp="${this._escape(entry.entry_id)}" ${entry.use_basic_dsp ? "checked" : ""}>
        Use basic audio cleanup for speech-to-text
      </label>
      <p class="muted">Applies light cleanup before speech-to-text. Voice recognition still uses the original audio.</p>
      <button data-save-stt="${this._escape(entry.entry_id)}" ${this._settingsBusy ? "disabled" : ""}>Save speech-to-text settings</button>
    </div>`).join("") : `<p class="muted">No Speaker Recognition speech-to-text service is configured.</p>`;

    const conversation = conversationEntries.length ? conversationEntries.map((entry, index) => `<div class="result" data-settings-entry="${this._escape(entry.entry_id)}">
      <strong>${this._escape(entry.title || `Conversation ${index + 1}`)}</strong>
      <label for="conversationEntity-${index}">Conversation agent</label>
      <select id="conversationEntity-${index}" data-conversation-entity="${this._escape(entry.entry_id)}">${this._entityOptions("conversation", entry.conversation_entity, entry.conversation_options)}</select>
      <details class="advancedSettings">
        <summary>Advanced identity setting</summary>
        <label for="confidence-${index}">Identity confidence: <span data-confidence-label="${this._escape(entry.entry_id)}">${Number(entry.min_confidence || 0).toFixed(2)}</span></label>
        <input id="confidence-${index}" type="range" min="0" max="1" step="0.05" value="${Number(entry.min_confidence || 0)}" data-confidence="${this._escape(entry.entry_id)}" style="width:100%;max-width:620px">
        <p class="muted">How confident recognition must be before Home Assistant uses the detected person\'s identity.</p>
      </details>
      <button data-save-conversation="${this._escape(entry.entry_id)}" ${this._settingsBusy ? "disabled" : ""}>Save conversation settings</button>
    </div>`).join("") : `<p class="muted">No Speaker Recognition conversation agent is configured.</p>`;

    return `<div class="card" id="settingsCard">
      <h2>Settings</h2>
      <p class="muted">Choose the speech-to-text and conversation services Speaker Recognition works with. Most users can leave Advanced settings unchanged.</p>
      ${this._settingsMessage ? `<div class="message">${this._escape(this._settingsMessage)}</div>` : ""}
      <h3>Speech-to-text</h3>
      ${stt}
      <h3>Conversation</h3>
      ${conversation}
      ${backend}
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
    if (card.classList.contains("enrollment-card")) return "enrollment";
    const heading = card.querySelector("h2");
    const title = heading ? heading.textContent.trim() : "";
    if (title === "Enroll or retrain a voice" || title === "Voice enrollment") return "enrollment";
    if (title === "Profile diagnostics" || title === "Profiles" || title === "Live satellite test" || title === "Recognition calibration") return "recognition";
    if (title === "Threshold guidance" || title === "Improve accuracy") return "improve";
    if (title === "Settings") return "settings";
    return "recognition";
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
      .profileName {
        display:inline-block;
        padding:6px 11px;
        border-radius:999px;
        background:var(--secondary-background-color);
        color:var(--primary-text-color);
        border:1px solid var(--divider-color);
      }
      button.profileName { cursor:pointer; }
      .profileName.selected {
        border:2px solid var(--primary-color);
        background:var(--card-background-color);
        color:var(--primary-color);
        font-weight:600;
      }
      .profileHealthStatus { margin:14px 0; padding:12px 14px; border-radius:9px; border-left:4px solid var(--divider-color); background:var(--secondary-background-color); }
      .profileHealthStatus p { margin:5px 0 0; }
      .profileHealthStatus.successStatus { border-left-color:var(--success-color, #43a047); }
      .profileHealthStatus.warningStatus { border-left-color:var(--warning-color, #ff9800); }
      .profileTechnicalDetails { margin-top:12px; }
      .profileHealthActions { margin-top:18px; }
      #settingsCard .result { border:1px solid var(--divider-color); background:transparent; }
      #calibrationGuidanceCard { background:var(--card-background-color) !important; }
      #calibrationGuidanceCard .result { background:transparent; border:1px solid var(--divider-color); }
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
          <span class="profileChip ${enrolled ? "ready" : ""}">${enrolled ? "✓ Voice profile active" : "No voice profile yet"}</span>
          ${staged.length ? `<span class="profileChip">Updating profile</span>` : ""}
          <span class="profileChip ${staged.length >= minimum ? "ready" : ""}">${staged.length} of ${minimum} needed</span>
        </div>
        <p class="muted">${enrolled
          ? (staged.length
            ? "Your existing voice profile keeps working until the updated profile is ready."
            : `To replace this profile, record at least ${minimum} of the ${total} phrases below.`)
          : (staged.length >= minimum ? "Enough recordings are ready to create this voice profile." : `Record at least ${minimum} of the ${total} phrases below.`)}</p>`;
    }

    const samples = card.querySelector(".samples");
    if (samples) {
      const progress = document.createElement("div");
      progress.className = "sampleProgress";
      progress.innerHTML = `<strong>Training phrases</strong><span>${staged.length} of ${minimum} needed</span>`;
      samples.insertAdjacentElement("beforebegin", progress);
      for (const button of samples.querySelectorAll("[data-sample]")) {
        const index = Number(button.dataset.sample);
        button.classList.toggle("active", index === this._sampleIndex);
        button.setAttribute("aria-current", index === this._sampleIndex ? "step" : "false");
        button.title = staged.includes(index)
          ? `Phrase ${index + 1}: recording ready`
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
      commit.textContent = enrolled ? "Update voice profile" : "Create voice profile";
      const guidance = commit.nextElementSibling;
      if (guidance) {
        guidance.textContent = remaining
          ? `${remaining} more recording${remaining === 1 ? "" : "s"} needed.`
          : `${staged.length} recording${staged.length === 1 ? "" : "s"} ready.`;
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
      const names = enrolled.map(userId => ({ userId, name: this._userName(userId) }));
      if (!this._profileHealthUserId && enrolled.length) this._profileHealthUserId = enrolled[0];
      paragraphs[0].innerHTML = `<strong>Enrolled voices</strong>${names.length
        ? `<div class="profileNames">${names.map(item => `<button class="profileName ${item.userId === this._profileHealthUserId ? "selected" : ""}" data-profile-health-user="${this._escape(item.userId)}" aria-pressed="${item.userId === this._profileHealthUserId ? "true" : "false"}">${this._escape(item.name)}</button>`).join("")}</div>`
        : `<span class="muted"> None</span>`}`;
    }
    if (paragraphs.length > 1) {
      paragraphs[1].innerHTML = "Select a voice above to check its profile, or test recognition below using one of your normal voice satellites.";
    }
    const health = document.createElement("div");
    health.innerHTML = `<h3>Profile health</h3>${this._renderProfileHealth()}<div class="profileHealthActions"><button id="refreshProfileHealth">Refresh profile health</button></div>`;
    card.appendChild(health);
    health.querySelector("#refreshProfileHealth").onclick = () => this._refreshProfileHealth();
    for (const button of card.querySelectorAll("[data-profile-health-user]")) {
      button.onclick = () => {
        this._profileHealthUserId = button.dataset.profileHealthUser;
        this._render();
      };
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
      ["enrollment", "Voices"],
      ["recognition", "Recognition"],
      ["improve", "Improve accuracy"],
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
        const similarity = this.shadowRoot.getElementById("backendSimilarity");
        const margin = this.shadowRoot.getElementById("backendMargin");
        if (!similarity.reportValidity() || !margin.reportValidity() || !similarity.value || !margin.value) return;
        const policy = { min_similarity: Number(similarity.value), min_margin: Number(margin.value) };
        const message = { entry_id: main.entry_id, backend_url: input.value };
        if (JSON.stringify(policy) !== JSON.stringify(main.acceptance_thresholds)) message.acceptance_thresholds = policy;
        this._saveSettings(message, main.entry_id);
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
