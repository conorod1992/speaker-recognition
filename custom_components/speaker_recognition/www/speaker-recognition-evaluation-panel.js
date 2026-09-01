import "./speaker-recognition-settings-panel.js";

const Panel = customElements.get("speaker-recognition-settings-panel");
const proto = Panel.prototype;
const baseRefreshHistory = proto._refreshHistory;
const basePanelSectionForCard = proto._panelSectionForCard;
const baseOrganizePanel = proto._organizePanel;
const baseBindPanelTabs = proto._bindPanelTabs;

proto._refreshHistory = async function(silent = false) {
  await baseRefreshHistory.call(this, silent);
  await this._refreshComparison(true);
};

proto._refreshComparison = async function(silent = false) {
  if (!this._hass) return;
  try {
    this._comparison = await this._call({ type: "speaker_recognition/shadow_comparison" });
    if (!silent) this._comparisonMessage = "";
  } catch (err) {
    this._comparisonMessage = this._errorText(err);
  }
  this._render();
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

proto._comparisonStateLabel = function(data) {
  const status = data && data.shadow_status ? data.shadow_status : {};
  if (!status.enabled) return { label: "Disabled", className: "comparison-disabled" };
  if (!status.ready) return { label: "Preparing profiles", className: "comparison-preparing" };
  if (data.ready) return { label: "Sufficient evidence", className: "comparison-ready" };
  return { label: "Collecting evidence", className: "comparison-preparing" };
};

proto._metricValue = function(engine, key) {
  if (!engine) return "—";
  if (key === "correct") return String((engine.correct_identity || 0) + (engine.correct_rejection || 0));
  if (key === "false_identifications") return String((engine.wrong_speaker || 0) + (engine.false_accepts || 0));
  if (key === "latency") {
    return engine.median_latency_seconds == null ? "—" : this._formatMs(engine.median_latency_seconds);
  }
  if (key === "similarity_threshold" || key === "margin_threshold") {
    return Number(engine[key] || 0).toFixed(2);
  }
  return String(engine[key] ?? "—");
};

proto._renderComparisonTable = function(data) {
  const authoritative = data.authoritative;
  const shadow = data.shadow;
  if (!authoritative || !shadow) {
    return `<div class="comparisonEmpty">
      <strong>No paired comparison metrics yet</strong>
      <p class="muted">Use Assist normally and label decisions in Recognition history. Once the same turn has both Resemblyzer and ECAPA scores, it counts as a paired trial.</p>
    </div>`;
  }

  const rows = [
    ["Correct decisions", "correct"],
    ["Wrong-speaker / false accepts", "false_identifications"],
    ["False unknowns", "false_unknowns"],
    ["Weighted error score", "score"],
    ["Best similarity threshold", "similarity_threshold"],
    ["Best margin threshold", "margin_threshold"],
    ["Median backend time", "latency"],
  ];
  return `<div class="comparisonTableWrap"><table class="comparisonTable">
    <thead><tr><th>Metric</th><th>${this._escape(this._formatEngineName(authoritative.engine_id))}<span>Active</span></th><th>${this._escape(this._formatEngineName(shadow.engine_id))}<span>Shadow</span></th></tr></thead>
    <tbody>${rows.map(([label, key]) => `<tr><th>${this._escape(label)}</th><td>${this._escape(this._metricValue(authoritative, key))}</td><td>${this._escape(this._metricValue(shadow, key))}</td></tr>`).join("")}</tbody>
  </table></div>`;
};

proto._renderComparisonCard = function() {
  const data = this._comparison;
  if (!data) {
    return `<div class="card" id="modelComparisonCard"><h2>Model evaluation</h2><p class="muted">Loading model comparison…</p></div>`;
  }

  const status = data.shadow_status || {};
  const state = this._comparisonStateLabel(data);
  const paired = Number(data.paired_count || 0);
  const minimum = Number(data.minimum_labelled || 15);
  const labelled = Number(data.labelled_count || 0);
  const coverage = Number(data.coverage || 0);
  const configuredCount = Array.isArray(status.configured_users) ? status.configured_users.length : 0;
  const enrolledCount = Array.isArray(status.enrolled_users) ? status.enrolled_users.length : 0;
  const progress = Math.min(100, minimum > 0 ? (paired / minimum) * 100 : 0);

  const enablement = status.enabled
    ? `<div class="comparisonStatusRow"><span class="comparisonBadge ${state.className}">${state.label}</span><span>${this._escape(this._formatEngineName(status.engine_id))} shadow evaluation</span></div>
       <p class="muted">Shadow profiles: ${enrolledCount} of ${configuredCount || enrolledCount} configured users ready.</p>`
    : `<div class="comparisonStatusRow"><span class="comparisonBadge comparison-disabled">Disabled</span><span>ECAPA shadow evaluation is not enabled.</span></div>
       <p class="muted">Set <code>shadow_engine: ecapa_tdnn</code> in the Speaker Recognition app options and restart the app to begin a side-by-side test.</p>`;

  return `<div class="card" id="modelComparisonCard">
    <div class="comparisonHeading"><div><h2>Model evaluation</h2><p class="muted">Compare Resemblyzer with ECAPA-TDNN on the exact same real Assist utterances.</p></div><button id="refreshComparisonBtn">Refresh</button></div>
    <div class="comparisonNotice"><strong>Resemblyzer remains authoritative.</strong> ECAPA results are experimental only and cannot change the speaker identity Home Assistant uses.</div>
    ${this._comparisonMessage ? `<div class="message">${this._escape(this._comparisonMessage)}</div>` : ""}
    ${enablement}
    <div class="comparisonProgress">
      <div><strong>${paired} / ${minimum} paired labelled decisions</strong><span>${labelled} total labelled · ${Math.round(coverage * 100)}% paired coverage</span></div>
      <div class="comparisonProgressTrack"><span style="width:${progress}%"></span></div>
    </div>
    ${paired > 0 && !data.ready ? `<p class="muted comparisonCaution">Results below are preliminary. Thresholds are optimized independently for each engine, but small datasets can overfit.</p>` : ""}
    ${this._renderComparisonTable(data)}
    <p class="muted comparisonFootnote">False identifications are weighted ${Number(data.false_identification_weight || 5)}× versus ${Number(data.false_unknown_weight || 1)}× for false unknowns. Lower weighted error is better; raw similarity values are not directly comparable between models.</p>
  </div>`;
};

proto._installComparisonStyles = function() {
  if (!this.shadowRoot || this.shadowRoot.getElementById("comparison-ui-style")) return;
  const style = document.createElement("style");
  style.id = "comparison-ui-style";
  style.textContent = `
    .comparisonHeading { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
    .comparisonHeading h2 { margin-bottom:4px; }
    .comparisonHeading p { margin-top:0; }
    .comparisonNotice { margin:14px 0; padding:12px 14px; border-radius:10px; border:1px solid var(--primary-color); background:var(--secondary-background-color); }
    .comparisonStatusRow { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-top:16px; font-weight:600; }
    .comparisonBadge { display:inline-flex; align-items:center; min-height:28px; padding:4px 9px; border:1px solid var(--divider-color); border-radius:999px; font-size:.88rem; }
    .comparison-ready { border-color:var(--success-color, #43a047); }
    .comparison-preparing { border-color:var(--warning-color, #ff9800); }
    .comparison-disabled { color:var(--secondary-text-color); }
    .comparisonProgress { margin:18px 0; }
    .comparisonProgress > div:first-child { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }
    .comparisonProgress span { color:var(--secondary-text-color); }
    .comparisonProgressTrack { height:8px; margin-top:9px; overflow:hidden; border-radius:999px; background:var(--secondary-background-color); }
    .comparisonProgressTrack span { display:block; height:100%; background:var(--primary-color); border-radius:999px; }
    .comparisonCaution { padding-left:12px; border-left:3px solid var(--warning-color, #ff9800); }
    .comparisonTableWrap { overflow-x:auto; margin-top:14px; }
    .comparisonTable { width:100%; min-width:560px; border-collapse:collapse; }
    .comparisonTable th, .comparisonTable td { padding:10px 12px; border-bottom:1px solid var(--divider-color); text-align:right; }
    .comparisonTable th:first-child { text-align:left; }
    .comparisonTable thead th { vertical-align:bottom; }
    .comparisonTable thead span { display:block; margin-top:2px; color:var(--secondary-text-color); font-size:.78rem; font-weight:500; }
    .comparisonEmpty { padding:16px; border:1px dashed var(--divider-color); border-radius:10px; }
    .comparisonEmpty p { margin-bottom:0; }
    .comparisonFootnote { margin-top:14px; font-size:.9rem; }
    @media (max-width: 700px) {
      .comparisonHeading { flex-direction:column; }
      .comparisonHeading button { width:100%; }
      .comparisonProgress > div:first-child { flex-direction:column; gap:2px; }
    }
  `;
  this.shadowRoot.append(style);
};

proto._organizePanel = function() {
  if (!this.shadowRoot || !this._status || !this._status.configured) return;
  const wrap = this.shadowRoot.querySelector(".wrap");
  if (!wrap) return;

  if (!wrap.querySelector("#modelComparisonCard")) {
    const holder = document.createElement("div");
    holder.innerHTML = this._renderComparisonCard();
    if (holder.firstElementChild) wrap.appendChild(holder.firstElementChild);
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
  const refresh = this.shadowRoot && this.shadowRoot.getElementById("refreshComparisonBtn");
  if (refresh) refresh.onclick = () => this._refreshComparison();
};
