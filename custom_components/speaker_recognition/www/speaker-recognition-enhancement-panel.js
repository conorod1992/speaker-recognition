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

Panel.prototype._renderEnhancementMetric = function(label, metrics) {
  if (!metrics || metrics.noise_floor_dbfs == null) return `<tr><td>${label}</td><td colspan="3">Unavailable</td></tr>`;
  return `<tr><td>${label}</td><td>${Number(metrics.noise_floor_dbfs).toFixed(1)} dBFS</td><td>${Number(metrics.speech_level_dbfs).toFixed(1)} dBFS</td><td>${Number(metrics.estimated_snr_db).toFixed(1)} dB</td></tr>`;
};

Panel.prototype._renderLiveResult = function(result) {
  let rendered = originalRenderLiveResult.call(this, result);
  if (!result || !rendered) return rendered;

  let comparison;
  if (this._enhancementSequence !== result.utterance_sequence || this._enhancementBusy) {
    comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">Preparing original, basic DSP, RNNoise-only and combined playback…</p></div>`;
  } else if (this._enhancementError) {
    comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">${this._escape(this._enhancementError)}</p></div>`;
  } else if (this._enhancementPreview) {
    const preview = this._enhancementPreview;
    const basicMs = Math.round(Number(preview.basic_processing_seconds ?? preview.processing_seconds ?? 0) * 1000);
    const rnnoiseMs = Math.round(Number(preview.rnnoise_processing_seconds || 0) * 1000);
    const comboMs = Math.round(Number(preview.neural_processing_seconds || 0) * 1000);
    const engine = this._escape(preview.neural_engine || "RNNoise");
    const rnnoisePlayer = preview.rnnoise_wav_base64
      ? `<div><b>RNNoise only</b><small>${engine} directly on the original HA audio</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.rnnoise_wav_base64}"></audio></div>`
      : `<div><b>RNNoise only</b><small>Direct neural denoise</small><p class="muted">${this._escape(preview.neural_error || "Neural preview unavailable")}</p></div>`;
    const comboPlayer = preview.neural_wav_base64
      ? `<div><b>Basic DSP + RNNoise</b><small>Current DSP preprocessing followed by ${engine}</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.neural_wav_base64}"></audio></div>`
      : `<div><b>Basic DSP + RNNoise</b><small>Combined path</small><p class="muted">${this._escape(preview.neural_error || "Neural preview unavailable")}</p></div>`;
    const metrics = preview.comparison_metrics || {};

    comparison = `<div class="result">
      <strong>Audio comparison</strong>
      <p class="muted">All four players use the same live Assist utterance. RNNoise-only now receives the untouched Home Assistant PCM, so this test separates RNNoise quality from any effect caused by the basic DSP preprocessing. Production STT is still unchanged.</p>
      <div class="audioCompare">
        <div><b>Original from Home Assistant</b><small>No extra processing</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.original_wav_base64}"></audio></div>
        <div><b>Basic DSP</b><small>High-pass, mains notches and conservative attenuation</small><audio controls preload="metadata" src="data:audio/wav;base64,${preview.enhanced_wav_base64}"></audio></div>
        ${rnnoisePlayer}
        ${comboPlayer}
      </div>
      <p class="muted">Basic DSP: ${basicMs} ms${preview.rnnoise_wav_base64 ? ` · RNNoise-only: ${rnnoiseMs} ms · Combined RNNoise stage: ${comboMs} ms` : ""} · Audio: ${Number(preview.audio_seconds || 0).toFixed(1)} s at ${preview.sample_rate} Hz.</p>
      <div class="metricTableWrap">
        <table class="metricTable">
          <thead><tr><th>Path</th><th>Estimated noise floor</th><th>Speech level</th><th>Estimated SNR</th></tr></thead>
          <tbody>
            ${this._renderEnhancementMetric("Original", metrics.original)}
            ${this._renderEnhancementMetric("Basic DSP", metrics.basic)}
            ${this._renderEnhancementMetric("RNNoise only", metrics.rnnoise)}
            ${this._renderEnhancementMetric("DSP + RNNoise", metrics.combo)}
          </tbody>
        </table>
      </div>
      <p class="muted">The noise/SNR figures are diagnostic estimates from quiet and speech-heavy 20 ms frames, not laboratory measurements. They are most useful for comparing the four versions of this same utterance.</p>
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
      .audioCompare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }
      .audioCompare > div { min-width: 0; }
      .audioCompare b, .audioCompare small { display: block; }
      .audioCompare small { margin-top: 4px; opacity: 0.72; }
      .audioCompare audio { display: block; width: 100%; margin-top: 8px; }
      .metricTableWrap { overflow-x: auto; margin-top: 12px; }
      .metricTable { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
      .metricTable th, .metricTable td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--divider-color); white-space: nowrap; }
      @media (max-width: 900px) { .audioCompare { grid-template-columns: 1fr; } }
    `;
    this.shadowRoot.append(style);
  }
};
