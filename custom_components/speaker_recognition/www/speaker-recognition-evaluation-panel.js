import "./speaker-recognition-settings-panel.js";

const Panel = customElements.get("speaker-recognition-settings-panel");
const proto = Panel.prototype;
const baseRefreshHistory = proto._refreshHistory;
const basePanelSectionForCard = proto._panelSectionForCard;
const baseOrganizePanel = proto._organizePanel;
const baseBindPanelTabs = proto._bindPanelTabs;
const baseDisconnected = proto.disconnectedCallback;

proto._refreshHistory = async function(silent = false) {
  await baseRefreshHistory.call(this, silent);
  await this._refreshComparison(true);
};

proto.disconnectedCallback = function() {
  if (this._evaluationPollTimer) clearTimeout(this._evaluationPollTimer);
  this._evaluationPollTimer = null;
  if (baseDisconnected) baseDisconnected.call(this);
};

proto._refreshComparison = async function(silent = false) {
  if (!this._hass) return;
  try {
    this._comparison = await this._call({ type: "speaker_recognition/evaluation_status" });
    if (!silent) this._comparisonMessage = "";
    const configured = this._comparison.shadow_status && this._comparison.shadow_status.configured_users;
    if (!this._evaluationGroundTruth && Array.isArray(configured) && configured.length) {
      this._evaluationGroundTruth = configured[0];
    }
  } catch (err) {
    this._comparisonMessage = this._errorText(err);
  }
  this._render();
  this._scheduleEvaluationPoll();
};

proto._scheduleEvaluationPoll = function() {
  if (this._evaluationPollTimer) clearTimeout(this._evaluationPollTimer);
  this._evaluationPollTimer = null;
  const data = this._comparison;
  if (!data || !data.running || data.pending) return;
  this._evaluationPollTimer = setTimeout(() => this._refreshComparison(true), 800);
};

proto._evaluationAction = async function(type, payload = {}) {
  if (!this._hass || this._evaluationBusy) return;
  this._evaluationBusy = true;
  this._comparisonMessage = "";
  this._render();
  try {
    this._comparison = await this._call({ type: `speaker_recognition/${type}`, ...payload });
  } catch (err) {
    this._comparisonMessage = this._errorText(err);
  } finally {
    this._evaluationBusy = false;
    this._render();
    this._scheduleEvaluationPoll();
  }
};

proto._panelSectionForCard = function(card) {
  if (card.id === "modelComparisonCard") return "evaluation";
  return basePanelSectionForCard.call(this, card);
};

proto._formatEngineName = function(engineId) {
  if (engineId === "resemblyzer") return "Resemblyzer";
  if (engineId === "ecapa_tdnn") return "ECAPA-TDNN";
  return engineId || "Unknown engine";
};

proto._evaluationUserName = function(userId) {
  if (!userId) return "Not enrolled";
  const users = this._status && Array.isArray(this._status.users) ? this._status.users : [];
  const match = users.find(user => user.id === userId);
  return match ? match.name : userId;
};

proto._evalMs = function(seconds) {
  return seconds == null ? "—" : `${Math.round(Number(seconds) * 1000)} ms`;
};

proto._pendingLatency = function(engine) {
  if (!engine || engine.effective_added_latency_seconds == null) return "—";
  const value = this._evalMs(engine.effective_added_latency_seconds);
  return engine.effective_added_latency_upper_bound ? `≤ ${value}` : value;
};

proto._aggregateLatency = function(engine) {
  if (!engine || engine.median_effective_added_latency_seconds == null) return "—";
  const value = this._evalMs(engine.median_effective_added_latency_seconds);
  return engine.effective_latency_contains_upper_bounds ? `~${value}*` : value;
};

proto._metricValue = function(engine, key) {
  if (!engine) return "—";
  if (key === "similarity_threshold") return Number(engine[key] || 0).toFixed(2);
  if (key === "margin_threshold") {
    return engine.margin_relevant ? Number(engine[key] || 0).toFixed(2) : "n/a";
  }
  if (key === "backend") return this._evalMs(engine.median_backend_seconds);
  if (key === "call") return this._evalMs(engine.median_call_seconds);
  if (key === "effective") return this._aggregateLatency(engine);
  return String(engine[key] ?? "—");
};

proto._renderComparisonTable = function(data) {
  const authoritative = data.authoritative;
  const shadow = data.shadow;
  if (!authoritative || !shadow) {
    return `<div class="comparisonEmpty">
      <strong>No saved A/B trials yet</strong>
      <p class="muted">Start testing, use Assist normally, then tell the evaluator who actually spoke. There is no trial limit; results remain here until you clear them.</p>
    </div>`;
  }

  const rows = [
    ["Correct decisions", "correct"],
    ["Correct enrolled-speaker IDs", "correct_identity"],
    ["Correct not-enrolled rejections", "correct_rejection"],
    ["Wrong-speaker / false accepts", "false_identifications"],
    ["False unknowns", "false_unknowns"],
    ["Weighted error score", "score"],
    ["Best similarity threshold", "similarity_threshold"],
    ["Best margin threshold", "margin_threshold"],
    ["Median backend time", "backend"],
    ["Median end-to-end model call", "call"],
    ["Median effective Assist latency", "effective"],
  ];
  return `<div class="comparisonTableWrap"><table class="comparisonTable">
    <thead><tr><th>Metric</th><th>${this._escape(this._formatEngineName(authoritative.engine_id))}<span>Active</span></th><th>${this._escape(this._formatEngineName(shadow.engine_id))}<span>Shadow</span></th></tr></thead>
    <tbody>${rows.map(([label, key]) => `<tr><th>${this._escape(label)}</th><td>${this._escape(this._metricValue(authoritative, key))}</td><td>${this._escape(this._metricValue(shadow, key))}</td></tr>`).join("")}</tbody>
  </table></div>`;
};

proto._renderPendingDiagnostics = function(data) {
  const pending = data.pending;
  if (!pending) return "";
  if (pending.authoritative_error || pending.shadow_error || !pending.authoritative || !pending.shadow) {
    const error = pending.authoritative_error || pending.shadow_error || "The paired score was incomplete.";
    return `<div class="evaluationPending">
      <h3>Trial could not be scored</h3>
      <p>${this._escape(error)}</p>
      <button id="evaluationDiscardBtn" class="secondary">Discard and wait for another</button>
    </div>`;
  }

  const auth = pending.authoritative;
  const shadow = pending.shadow;
  const diagnostics = [
    ["Candidate", this._evaluationUserName(auth.candidate_user_id), this._evaluationUserName(shadow.candidate_user_id)],
    ["Raw similarity", Number(auth.similarity).toFixed(3), Number(shadow.similarity).toFixed(3)],
    ["Raw margin", auth.margin == null ? "n/a" : Number(auth.margin).toFixed(3), shadow.margin == null ? "n/a" : Number(shadow.margin).toFixed(3)],
    ["Backend time", this._evalMs(auth.backend_processing_seconds), this._evalMs(shadow.backend_processing_seconds)],
    ["End-to-end model call", this._evalMs(auth.call_seconds), this._evalMs(shadow.call_seconds)],
    ["Effective Assist latency", this._pendingLatency(auth), this._pendingLatency(shadow)],
  ];
  const configured = data.shadow_status && Array.isArray(data.shadow_status.configured_users)
    ? data.shadow_status.configured_users : [];
  const stt = pending.stt_seconds == null ? "" : `<span>STT: ${this._escape(this._evalMs(pending.stt_seconds))}</span>`;

  let groundTruth = "";
  if (configured.length === 1) {
    const userId = configured[0];
    const name = this._evaluationUserName(userId);
    groundTruth = `<div class="groundTruth"><strong>Was that ${this._escape(name)}?</strong><div class="evaluationButtons">
      <button data-evaluation-user="${this._escape(userId)}">Yes — ${this._escape(name)}</button>
      <button data-evaluation-user="__unknown__" class="secondary">No — someone not enrolled</button>
      <button id="evaluationDiscardBtn" class="secondary">Discard trial</button>
    </div></div>`;
  } else {
    groundTruth = `<div class="groundTruth"><strong>Who actually spoke?</strong><div class="evaluationButtons">
      <select id="evaluationGroundTruthSelect">
        ${configured.map(userId => `<option value="${this._escape(userId)}" ${this._evaluationGroundTruth === userId ? "selected" : ""}>${this._escape(this._evaluationUserName(userId))}</option>`).join("")}
        <option value="__unknown__" ${this._evaluationGroundTruth === "__unknown__" ? "selected" : ""}>Someone not enrolled</option>
      </select>
      <button id="evaluationSaveGroundTruthBtn">Save ground truth</button>
      <button id="evaluationDiscardBtn" class="secondary">Discard trial</button>
    </div></div>`;
  }

  return `<div class="evaluationPending">
    <div class="pendingHeading"><div><h3>Utterance scored</h3><p class="muted">Both engines heard the exact same buffered Assist audio.</p></div>${stt}</div>
    <div class="comparisonTableWrap"><table class="comparisonTable pendingTable">
      <thead><tr><th>Diagnostic</th><th>${this._escape(this._formatEngineName(auth.engine_id))}</th><th>${this._escape(this._formatEngineName(shadow.engine_id))}</th></tr></thead>
      <tbody>${diagnostics.map(row => `<tr><th>${this._escape(row[0])}</th><td>${this._escape(row[1])}</td><td>${this._escape(row[2])}</td></tr>`).join("")}</tbody>
    </table></div>
    <p class="muted latencyNote">Effective Assist latency accounts for STT and recognition running in parallel. A ≤ value is a conservative upper bound on turns where STT was still running after the current analysis finished.</p>
    ${groundTruth}
  </div>`;
};

proto._evaluationState = function(data) {
  const status = data && data.shadow_status ? data.shadow_status : {};
  if (!status.enabled) return { label: "Disabled", className: "comparison-disabled", text: "ECAPA shadow evaluation is not enabled." };
  if (!status.ready) return { label: "Preparing profiles", className: "comparison-preparing", text: "Wait for the shadow profile to finish preparing before starting a test." };
  if (data.pending) return { label: "Ground truth needed", className: "comparison-preparing", text: "Review the paired diagnostics below and tell the evaluator who actually spoke." };
  if (data.scoring) return { label: "Scoring utterance", className: "comparison-preparing", text: "The same Assist audio is being scored by both engines." };
  if (data.running) return { label: "Waiting for utterance", className: "comparison-ready", text: "Use Assist normally. The next Speaker Recognition utterance will become a test trial." };
  return { label: "Ready", className: "comparison-ready", text: "Start testing when you are ready to collect explicitly labelled A/B trials." };
};

proto._renderComparisonCard = function() {
  const data = this._comparison;
  if (!data) {
    return `<div class="card" id="modelComparisonCard"><h2>Model evaluation</h2><p class="muted">Loading live model evaluation…</p></div>`;
  }
  const status = data.shadow_status || {};
  const state = this._evaluationState(data);
  const configuredCount = Array.isArray(status.configured_users) ? status.configured_users.length : 0;
  const enrolledCount = Array.isArray(status.enrolled_users) ? status.enrolled_users.length : 0;
  const trials = Number(data.trial_count || 0);
  const known = Number(data.known_trials || 0);
  const unknown = Number(data.unknown_trials || 0);
  const controlsDisabled = this._evaluationBusy ? "disabled" : "";

  return `<div class="card" id="modelComparisonCard">
    <div class="comparisonHeading"><div><h2>Model evaluation</h2><p class="muted">Run Resemblyzer and ECAPA-TDNN against the exact same real Assist utterances, then supply independent ground truth.</p></div><div class="evaluationTopButtons"><button id="refreshComparisonBtn" class="secondary" ${controlsDisabled}>Refresh</button><button id="evaluationStartStopBtn" ${!status.ready || this._evaluationBusy ? "disabled" : ""}>${data.running ? "Stop testing" : "Start testing"}</button></div></div>
    <div class="comparisonNotice"><strong>Resemblyzer remains authoritative.</strong> ECAPA runs beside it only for measurement and never changes the identity Home Assistant uses.</div>
    ${this._comparisonMessage ? `<div class="message">${this._escape(this._comparisonMessage)}</div>` : ""}
    <div class="comparisonStatusRow"><span class="comparisonBadge ${state.className}">${this._escape(state.label)}</span><span>${this._escape(state.text)}</span></div>
    ${status.enabled ? `<p class="muted">Shadow profiles: ${enrolledCount} of ${configuredCount || enrolledCount} configured users ready.</p>` : `<p class="muted">Set <code>shadow_engine: ecapa_tdnn</code> in the Speaker Recognition app options and restart the app.</p>`}
    ${this._renderPendingDiagnostics(data)}
    <div class="savedTrialsHeading"><div><h3>Saved results</h3><p class="muted">${trials} labelled A/B trial${trials === 1 ? "" : "s"}${trials ? ` · ${known} enrolled speaker · ${unknown} not enrolled` : ""}. Results keep accumulating until you clear them.</p></div><button id="evaluationClearBtn" class="secondary" ${trials === 0 || this._evaluationBusy ? "disabled" : ""}>Clear results</button></div>
    ${this._renderComparisonTable(data)}
    <p class="muted comparisonFootnote">False identifications are weighted ${Number(data.false_identification_weight || 5)}× versus ${Number(data.false_unknown_weight || 1)}× for false unknowns. Thresholds are optimized independently because raw similarity values are not comparable between models. ${data.authoritative && data.authoritative.effective_latency_contains_upper_bounds || data.shadow && data.shadow.effective_latency_contains_upper_bounds ? "*The latency median includes one or more conservative upper-bound estimates." : ""}</p>
  </div>`;
};

proto._installComparisonStyles = function() {
  if (!this.shadowRoot || this.shadowRoot.getElementById("comparison-ui-style")) return;
  const style = document.createElement("style");
  style.id = "comparison-ui-style";
  style.textContent = `
    .comparisonHeading, .savedTrialsHeading, .pendingHeading { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
    .comparisonHeading h2, .savedTrialsHeading h3, .pendingHeading h3 { margin-bottom:4px; }
    .comparisonHeading p, .savedTrialsHeading p, .pendingHeading p { margin-top:0; }
    .evaluationTopButtons, .evaluationButtons { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
    .comparisonNotice { margin:14px 0; padding:12px 14px; border-radius:10px; border:1px solid var(--primary-color); background:var(--secondary-background-color); }
    .comparisonStatusRow { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:16px; font-weight:600; }
    .comparisonBadge { display:inline-flex; align-items:center; min-height:28px; padding:4px 9px; border:1px solid var(--divider-color); border-radius:999px; font-size:.88rem; }
    .comparison-ready { border-color:var(--success-color, #43a047); }
    .comparison-preparing { border-color:var(--warning-color, #ff9800); }
    .comparison-disabled { color:var(--secondary-text-color); }
    .evaluationPending { margin:18px 0; padding:16px; border:1px solid var(--divider-color); border-radius:12px; background:var(--secondary-background-color); }
    .groundTruth { margin-top:16px; padding-top:14px; border-top:1px solid var(--divider-color); }
    .groundTruth strong { display:block; margin-bottom:10px; }
    .comparisonTableWrap { overflow-x:auto; margin-top:14px; }
    .comparisonTable { width:100%; min-width:600px; border-collapse:collapse; }
    .comparisonTable th, .comparisonTable td { padding:10px 12px; border-bottom:1px solid var(--divider-color); text-align:right; }
    .comparisonTable th:first-child { text-align:left; }
    .comparisonTable thead th { vertical-align:bottom; }
    .comparisonTable thead span { display:block; margin-top:2px; color:var(--secondary-text-color); font-size:.78rem; font-weight:500; }
    .pendingTable { min-width:520px; }
    .comparisonEmpty { padding:16px; border:1px dashed var(--divider-color); border-radius:10px; }
    .comparisonEmpty p { margin-bottom:0; }
    .savedTrialsHeading { margin-top:22px; align-items:center; }
    .comparisonFootnote, .latencyNote { margin-top:14px; font-size:.9rem; }
    @media (max-width: 700px) {
      .comparisonHeading, .savedTrialsHeading, .pendingHeading { flex-direction:column; }
      .evaluationTopButtons, .evaluationButtons { width:100%; }
      .evaluationTopButtons button, .evaluationButtons button, .evaluationButtons select, .savedTrialsHeading button { flex:1 1 auto; }
    }
  `;
  this.shadowRoot.append(style);
};

proto._organizePanel = function() {
  if (!this.shadowRoot || !this._status || !this._status.configured) return;
  const wrap = this.shadowRoot.querySelector(".wrap");
  if (!wrap) return;

  const existing = wrap.querySelector("#modelComparisonCard");
  const holder = document.createElement("div");
  holder.innerHTML = this._renderComparisonCard();
  if (holder.firstElementChild) {
    if (existing) existing.replaceWith(holder.firstElementChild);
    else wrap.appendChild(holder.firstElementChild);
  }

  baseOrganizePanel.call(this);
  this._installComparisonStyles();

  const tabs = this.shadowRoot.querySelector(".panelTabs");
  if (tabs && !tabs.querySelector('[data-panel-tab="evaluation"]')) {
    const button = document.createElement("button");
    button.className = `panelTab ${this._panelSection === "evaluation" ? "active" : ""}`;
    button.dataset.panelTab = "evaluation";
    button.setAttribute("role", "tab");
    button.setAttribute("aria-selected", this._panelSection === "evaluation" ? "true" : "false");
    button.textContent = "Evaluation";
    const settings = tabs.querySelector('[data-panel-tab="settings"]');
    tabs.insertBefore(button, settings || null);
  }

  const card = wrap.querySelector("#modelComparisonCard");
  if (card) {
    card.dataset.panelSection = "evaluation";
    card.hidden = this._panelSection !== "evaluation";
  }
};

proto._bindPanelTabs = function() {
  baseBindPanelTabs.call(this);
  const $ = id => this.shadowRoot && this.shadowRoot.getElementById(id);
  if ($("refreshComparisonBtn")) $("refreshComparisonBtn").onclick = () => this._refreshComparison();
  if ($("evaluationStartStopBtn")) $("evaluationStartStopBtn").onclick = () => this._evaluationAction(this._comparison && this._comparison.running ? "evaluation_stop" : "evaluation_start");
  if ($("evaluationClearBtn")) $("evaluationClearBtn").onclick = () => {
    if (window.confirm("Clear all saved model evaluation results?")) this._evaluationAction("evaluation_clear");
  };
  if ($("evaluationDiscardBtn")) $("evaluationDiscardBtn").onclick = () => this._evaluationAction("evaluation_discard");
  if ($("evaluationGroundTruthSelect")) $("evaluationGroundTruthSelect").onchange = event => { this._evaluationGroundTruth = event.target.value; };
  if ($("evaluationSaveGroundTruthBtn")) $("evaluationSaveGroundTruthBtn").onclick = () => {
    const value = this._evaluationGroundTruth === "__unknown__" ? null : this._evaluationGroundTruth;
    this._evaluationAction("evaluation_label", { actual_user_id: value });
  };
  for (const button of this.shadowRoot.querySelectorAll("[data-evaluation-user]")) {
    button.onclick = () => {
      const value = button.dataset.evaluationUser === "__unknown__" ? null : button.dataset.evaluationUser;
      this._evaluationAction("evaluation_label", { actual_user_id: value });
    };
  }
};
