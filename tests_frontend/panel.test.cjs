const { test } = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const { JSDOM } = require("jsdom");

function setup(t, evaluation = false) {
  const dom = new JSDOM("<!doctype html><body></body>", { runScripts: "outside-only", url: "https://ha.example/" });
  const w = dom.window;
  w.CSS = { escape: value => value };
  w.URL.revokeObjectURL = () => {};
  const files = ["panel", "calibration-panel", "enhancement-panel", "settings-panel"];
  if (evaluation) files.push("evaluation-panel");
  for (const file of files) {
    const source = readFileSync(join(__dirname, "../custom_components/speaker_recognition/www", `speaker-recognition-${file}.js`), "utf8");
    w.eval(`{${source.replace(/^import .*;\r?\n/gm, "").replace(/^import\(.*\);\r?\n?/gm, "")}\n}`);
  }
  const p = w.document.createElement("speaker-recognition-settings-panel");
  w.document.body.append(p);
  p._status = {
    configured: true, users: [{ id: "alice", name: "Alice" }, { id: "bob", name: "Bob" }],
    enrolled_users: ["alice", "bob"], staged: { alice: [0, 1, 2, 4] },
    satellites: [], enrollment_satellites: [], phrases: ["A", "B", "C", "D", "E", "F"], minimum_samples: 5,
  };
  p._userId = "alice";
  p._feedbackUserId = "alice";
  p._settings = {
    main: { entry_id: "main", backend_url: "http://backend:8099", acceptance_thresholds: { min_similarity: .55, min_margin: .05 } },
    stt_entries: [], conversation_entries: [{ entry_id: "conversation", conversation_entity: "conversation.source", conversation_options: ["conversation.source"], min_confidence: .73 }],
  };
  p._render();
  t.after(() => { p.remove(); w.close(); });
  return { p, w };
}

function media(w, p) {
  p._canUseMicrophone = () => true;
  const pending = [], tracks = [], contexts = [];
  Object.defineProperty(w.navigator, "mediaDevices", { configurable: true, value: {
    getUserMedia: () => new Promise((resolve, reject) => pending.push({ resolve, reject })),
  } });
  w.AudioContext = class {
    constructor() { this.sampleRate = 16000; this.destination = {}; this.closed = false; contexts.push(this); }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { return { connect() {}, disconnect() {} }; }
    close() { this.closed = true; return Promise.resolve(); }
  };
  return { pending, tracks, contexts, grant(index = 0) {
    const track = { stopped: false, stop() { this.stopped = true; } };
    tracks.push(track);
    pending[index].resolve({ getTracks: () => [track] });
  } };
}

test("registered panel serializes permission acquisition and stops all resources", async t => {
  const { p, w } = setup(t), m = media(w, p);
  const start = p._startRecording();
  assert.equal(p.shadowRoot.getElementById("recordBtn").disabled, true);
  await p._startRecording();
  assert.equal(m.pending.length, 1);
  m.grant();
  await start;
  await p._stopRecording();
  assert.equal(m.tracks[0].stopped, true);
  assert.equal(m.contexts[0].closed, true);
});

test("permission granted after navigation is immediately released", async t => {
  const { p, w } = setup(t), m = media(w, p);
  const start = p._startRecording();
  p.remove();
  m.grant();
  await start;
  assert.equal(p._recording, null);
  assert.equal(m.tracks[0].stopped, true);
  assert.equal(m.contexts.length, 0);
});

test("partial audio initialization failure releases the acquired stream", async t => {
  const { p, w } = setup(t), m = media(w, p);
  w.AudioContext.prototype.createScriptProcessor = () => { throw new Error("processor failed"); };
  const start = p._startRecording();
  m.grant();
  await start;
  assert.equal(m.tracks[0].stopped, true);
  assert.equal(m.contexts[0].closed, true);
  assert.equal(p._recordingPending, false);
});

for (const failure of ["short", "denied"]) {
  test(`${failure} retake cannot reuse the previous recording`, async t => {
    const { p, w } = setup(t), m = media(w, p);
    p._lastWav = new Uint8Array([1, 2, 3]);
    const start = p._startRecording();
    assert.equal(p._lastWav, null);
    if (failure === "denied") m.pending[0].reject(new Error("denied"));
    else m.grant();
    await start;
    if (failure === "short") await p._stopRecording();
    assert.equal(p.shadowRoot.getElementById("useBtn"), null);
    assert.equal(p._lastWav, null);
  });
}

test("feedback selector and submitted ground truth agree after rerender", async t => {
  const { p, w } = setup(t);
  const select = p.shadowRoot.getElementById("feedbackUserSelect");
  select.value = "__unknown__";
  select.dispatchEvent(new w.Event("change", { bubbles: true }));
  p._render();
  assert.equal(p.shadowRoot.getElementById("feedbackUserSelect").value, "__unknown__");
  const sent = [];
  p._call = async message => { sent.push(message); return {}; };
  p._refreshHistory = async () => {};
  await p._submitReviewFeedback("decision", "wrong_speaker", "__selected__");
  assert.equal(sent[0].actual_user_id, null);
});

test("quality refresh preserves settings draft, caret, focus, and disclosure", t => {
  const { p, w } = setup(t);
  p._panelSection = "settings";
  p._render();
  const input = p.shadowRoot.getElementById("backendUrl");
  input.closest("details").open = true;
  input.value = "http://changed:8099";
  input.dispatchEvent(new w.Event("input", { bubbles: true }));
  input.focus();
  input.setSelectionRange(4, 7);
  p._render();
  const updated = p.shadowRoot.getElementById("backendUrl");
  assert.equal(updated.value, "http://changed:8099");
  assert.equal(updated.closest("details").open, true);
  assert.equal(p.shadowRoot.activeElement, updated);
  assert.equal(updated.selectionStart, 4);
});

test("successful save clears only the saved entry's draft", async t => {
  const { p, w } = setup(t);
  const input = p.shadowRoot.getElementById("backendUrl");
  input.value = "http://changed:8099";
  input.dispatchEvent(new w.Event("input", { bubbles: true }));
  p._settingsDrafts.set("conversation:data-confidence", ".81");
  p._call = async () => ({});
  p._refreshSettings = async () => { p._settings.main.backend_url = "http://changed:8099"; };
  await p._saveSettings({ backend_url: input.value }, "main");
  assert.equal(p._settingsDrafts.has("main:backendUrl"), false);
  assert.equal(p._settingsDrafts.has("conversation:data-confidence"), true);
});

test("review audio node and playback position survive refresh", t => {
  const { p } = setup(t);
  p._history = { decisions: [{ decision_id: "d", has_audio: true, user_id: "alice", candidate_user_id: "alice", identity_eligible: true }] };
  p._reviewAudioUrls.set("d", "blob:test-audio");
  p._render();
  const audio = p.shadowRoot.querySelector("audio");
  audio.currentTime = 1.25;
  p._render();
  assert.equal(p.shadowRoot.querySelector("audio"), audio);
  assert.equal(audio.currentTime, 1.25);
});

test("satellite completion advances to next incomplete phrase in registered subclass", async t => {
  const { p, w } = setup(t);
  let poll;
  w.setTimeout = callback => { poll = callback; return 1; };
  p._refresh = async () => {};
  p._sampleIndex = 0;
  p._status.completed_satellite_captures = ["session"];
  p._pollForSatelliteCapture(0, "session");
  await poll();
  assert.equal(p._sampleIndex, 3);
});

test("initial status failure exposes the error and a working retry", async t => {
  const { p } = setup(t);
  const status = p._status;
  p._status = null;
  p._hass = { states: {} };
  p._call = async () => { throw new Error("Disconnected from Home Assistant"); };
  await p._refresh();
  assert.match(p.shadowRoot.querySelector('[role="alert"]').textContent, /Disconnected/);
  p._call = async () => status;
  p._refreshHistory = async () => {};
  p.shadowRoot.getElementById("retryStatusBtn").click();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(p._status, status);
});

test("tabs support keyboard navigation and label their visible panel", t => {
  const { p, w } = setup(t);
  const first = p.shadowRoot.querySelector('[role="tab"][tabindex="0"]');
  first.dispatchEvent(new w.KeyboardEvent("keydown", { key: "End", bubbles: true }));
  const active = p.shadowRoot.querySelector('[role="tab"][tabindex="0"]');
  assert.equal(active.dataset.panelTab, "settings");
  assert.equal(p.shadowRoot.activeElement, active);
  const region = p.shadowRoot.getElementById(active.getAttribute("aria-controls"));
  assert.equal(region.hidden, false);
  assert.equal(region.getAttribute("aria-labelledby"), active.id);
  assert.equal(p.shadowRoot.getElementById("backendUrl").closest('[role="tabpanel"]'), region);
  const slider = p.shadowRoot.querySelector("[data-confidence]");
  assert.equal(slider.value, "0.73");
  assert.equal(slider.validity.stepMismatch, false);
});

test("optional evaluation tab also participates in accessible navigation", t => {
  const { p } = setup(t, true);
  const tab = p.shadowRoot.querySelector('[data-panel-tab="evaluation"]');
  tab.click();
  assert.equal(p.shadowRoot.getElementById("panel-evaluation").hidden, false);
  assert.equal(p.shadowRoot.getElementById("tab-evaluation").tabIndex, 0);
});
