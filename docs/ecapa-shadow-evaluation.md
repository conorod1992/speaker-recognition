# Experimental ECAPA shadow evaluation

The application can optionally run SpeechBrain ECAPA-TDNN as a non-authoritative shadow engine while Resemblyzer continues to make every Home Assistant identity decision.

## Enable it

For the Home Assistant app, set `shadow_engine` to `ecapa_tdnn` and restart the app. Leave it at the default `none` for normal operation. The first shadow training run downloads the pinned `speechbrain/spkrec-ecapa-voxceleb` model into the configured model cache; subsequent starts reuse that cache.

For Docker Compose, set `SHADOW_ENGINE=ecapa_tdnn` before starting the service.

## Safety and performance behavior

- Resemblyzer remains the authoritative `/recognize` engine.
- ECAPA uses separate persisted profiles under `embeddings/shadow/ecapa_tdnn`.
- Existing enrollment WAVs are reused to build shadow profiles; users do not need to re-enroll.
- Missing/outdated shadow profiles are rebuilt in a background Home Assistant task.
- ECAPA scoring runs after the normal Assist recognition event in a separate task and does not delay the Assist response.
- Shadow download/training/scoring failures are best-effort and do not make `/health` unhealthy or remove authoritative profiles.
- No audio or transcript is added to decision history. Only engine ID, per-user scores, margins, and timing are persisted alongside the existing labelled turn.

## Evaluate results

The integration exposes the admin WebSocket command `speaker_recognition/shadow_comparison`. Once at least 15 paired decisions have been labelled in the normal Recognition Calibration history, it reports independently optimized similarity/margin operating points and open-set error counts for Resemblyzer and ECAPA. Raw score values are deliberately never compared using a shared threshold because the two embedding spaces have different score distributions.

This mode is intended for evaluation only. A later change can expose an engine selector if real Home Assistant hardware results justify making ECAPA authoritative.
