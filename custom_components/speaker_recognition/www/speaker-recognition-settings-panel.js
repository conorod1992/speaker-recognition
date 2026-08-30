import "./speaker-recognition-calibration-panel.js";

const BasePanel = customElements.get("speaker-recognition-calibration-panel");

class SpeakerRecognitionSettingsPanel extends BasePanel {
  constructor() {
    super();
    this._settings = null;
    this._settingsMessage = "";
    this._settingsBusy = "";
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
      return `<div class="card"><h2>Settings</h2><p class="muted">Loading settings…</p></div>`;
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

  _render() {
    super._render();
    if (!this.shadowRoot || !this._status || !this._status.configured) return;
    const wrap = this.shadowRoot.querySelector(".wrap");
    if (!wrap) return;
    const holder = document.createElement("div");
    holder.innerHTML = this._renderSettingsCard();
    const card = holder.firstElementChild;
    if (card) wrap.appendChild(card);
    this._bindSettingsEvents();
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
