import "./speaker-recognition-calibration-panel.js";

const BasePanel = customElements.get("speaker-recognition-calibration-panel");

class SpeakerRecognitionEnhancementPanel extends BasePanel {
  constructor() {
    super();
    this._enhancementPreview = null;
    this._enhancementSequence = null;
    this._enhancementBusy = false;
    this._enhancementError = "";
  }

  async _refresh(silent = false) {
    await super._refresh(silent);
    const result = this._status && this._status.live_test_result;
    const sequence = result && result.utterance_sequence;
    if (sequence && sequence !== this._enhancementSequence && !this._enhancementBusy) {
      await this._loadEnhancementPreview(sequence);
    }
  }

  async _loadEnhancementPreview(sequence) {
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
  }

  _renderLiveResult(result) {
    let rendered = super._renderLiveResult(result);
    if (!result || !rendered) return rendered;

    let comparison;
    if (this._enhancementSequence !== result.utterance_sequence || this._enhancementBusy) {
      comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">Preparing original and enhanced playback…</p></div>`;
    } else if (this._enhancementError) {
      comparison = `<div class="result"><strong>Audio comparison</strong><p class="muted">${this._escape(this._enhancementError)}</p></div>`;
    } else if (this._enhancementPreview) {
      const preview = this._enhancementPreview;
      const processingMs = Math.round(Number(preview.processing_seconds || 0) * 1000);
      comparison = `<div class="result">
        <strong>Audio comparison</strong>
        <p class="muted">These are the same live Assist utterance after Home Assistant delivered it to the Speaker Recognition wrapper. The enhanced version is an experimental dependency-free speech cleanup preview; production STT is unchanged in this v1.</p>
        <div class="audioCompare">
          <div><b>Original from Home Assistant</b><audio controls preload="metadata" src="data:audio/wav;base64,${preview.original_wav_base64}"></audio></div>
          <div><b>Enhanced preview</b><audio controls preload="metadata" src="data:audio/wav;base64,${preview.enhanced_wav_base64}"></audio></div>
        </div>
        <p class="muted">Enhancement processing: ${processingMs} ms for ${Number(preview.audio_seconds || 0).toFixed(1)} s of audio at ${preview.sample_rate} Hz.</p>
      </div>`;
    } else {
      comparison = "";
    }

    const closingIndex = rendered.lastIndexOf("</div>");
    if (closingIndex >= 0 && comparison) {
      rendered = `${rendered.slice(0, closingIndex)}${comparison}${rendered.slice(closingIndex)}`;
    }
    return rendered;
  }

  _render() {
    super._render();
    if (!this.shadowRoot) return;
    if (!this.shadowRoot.getElementById("enhancement-style")) {
      const style = document.createElement("style");
      style.id = "enhancement-style";
      style.textContent = `
        .audioCompare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 12px; }
        .audioCompare > div { min-width: 0; }
        .audioCompare audio { display: block; width: 100%; margin-top: 8px; }
        @media (max-width: 700px) { .audioCompare { grid-template-columns: 1fr; } }
      `;
      this.shadowRoot.append(style);
    }
  }
}

customElements.define("speaker-recognition-enhancement-panel", SpeakerRecognitionEnhancementPanel);
