import "./speaker-recognition-calibration-panel.js";

const Panel = customElements.get("speaker-recognition-calibration-panel");
const originalRefresh = Panel.prototype._refresh;
const originalRenderLiveResult = Panel.prototype._renderLiveResult;
const originalRender = Panel.prototype._render;

Panel.prototype._refresh = async function(silent = false) {
  await originalRefresh.call(this, silent);
  const result = this._status && this._status.live_test_result;
  const sequence = result && result.utterance_sequence;
  if (sequence && sequence !== this._enhancementSequence && !this._enhancementBusy) {
    await this._loadEnhancementPreview(sequence);
  }
};

Panel.prototype._loadEnhancementPreview = async function(sequence) {
  this._enhancementBusy = true;
  this._enhancementError = "";
  this._enhancementPreview = null;
  this._enhancementSequence = sequence;
  this._render();
  try {
    this._enhancementPreview = await this._call({
      type: "speaker_recognition/enhancement_preview",
      utterance_sequence: sequence,
    });
  } catch (err) {
    this._enhancementError = this._errorText(err);
  } finally {
    this._enhancementBusy = false;
    this._render();
  }
};

Panel.prototype._renderLiveResult = function(result) {
  let rendered = originalRenderLiveResult.call(this, result);
  if (!result || !rendered) return rendered;

  let comparison;
  if (this._enhancementSequence !== result.utterance_sequence || this._enhancementBusy) {
    comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">Preparing original, basic DSP and neural-denoise playback…</p></div>`;
  } else if (this._enhancementError) {
    comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">${this._escape(this._enhancementError)}</p></div>`;
  } else if (this._enhancementPreview) {
    const preview = this._enhancementPreview;
    const basicMs = Math.round(Number(preview.basic_processing_seconds ?? preview.processing_seconds ?? 0) * 1000);
    const neuralMs = Math.round(Number(preview.neural_processing_seconds || 0) * 1000);
    const neuralPlayer = preview.neural_wav_base64
      ? `<div><b>Neural denoise</b><small>Basic DSP + ${this._escape(preview.neural_engine || "RNNoise")}</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.neural_wav_base64}"></audio></div>`
      : `<div><b>Neural denoise</b><small>Basic DSP + RNNoise</small><p class="muted">${this._escape(preview.neural_error || "Neural preview unavailable")}</p></div>`;

    comparison = `<div class="result">
      <strong>Audio comparison</strong>
      <p class="muted">All three players use the same live Assist utterance. Basic DSP is the current lightweight cleanup. Neural denoise runs that result through RNNoise in the Speaker Recognition backend. Production STT is still unchanged while we compare quality and latency.</p>
      <div class="audioCompare">
        <div><b>Original from Home Assistant</b><small>No extra processing</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.original_wav_base64}"></audio></div>
        <div><b>Basic DSP</b><small>High-pass, mains notches and conservative attenuation</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.enhanced_wav_base64}"></audio></div>
        ${neuralPlayer}
      </div>
      <p class="muted">Basic DSP: ${basicMs} ms${preview.neural_wav_base64 ? ` · Neural stage: ${neuralMs} ms` : ""} · Audio: ${Number(preview.audio_seconds || 0).toFixed(1)} s at ${preview.sample_rate} Hz.</p>
    </div>`;
  } else {
    comparison = "";
  }

  const closingIndex = rendered.lastIndexOf("</div>");
  if (closingIndex >= 0 && comparison) {
    rendered = `${rendered.slice(0, closingIndex)}${comparison}${rendered.slice(closingIndex)}`;
  }
  return rendered;
};

Panel.prototype._render = function() {
  if (this._enhancementBusy == null) this._enhancementBusy = false;
  if (this._enhancementError == null) this._enhancementError = "";
  if (this._enhancementPreview == null) this._enhancementPreview = null;
  if (this._enhancementSequence == null) this._enhancementSequence = null;

  originalRender.call(this);
  if (!this.shadowRoot) return;
  if (!this.shadowRoot.getElementById("enhancement-style")) {
    const style = document.createElement("style");
    style.id = "enhancement-style";
    style.textContent = `
      .audioCompare { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }
      .audioCompare > div { min-width: 0; }
      .audioCompare b, .audioCompare small { display: block; }
      .audioCompare small { margin-top: 4px; opacity: 0.72; }
      .audioCompare audio { display: block; width: 100%; margin-top: 8px; }
      @media (max-width: 900px) { .audioCompare { grid-template-columns: 1fr; } }
    `;
    this.shadowRoot.append(style);
  }
};
