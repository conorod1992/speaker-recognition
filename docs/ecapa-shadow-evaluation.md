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

Open **Speaker Recognition → Evaluation** in the Home Assistant sidebar panel. The page shows whether ECAPA is disabled, preparing profiles, collecting evidence, or has sufficient evidence; how many labelled decisions have paired Resemblyzer/ECAPA scores; paired-data coverage; and the two engines side by side.

The comparison reports correct decisions, wrong-speaker/false accepts, false unknowns, a weighted error score, independently optimized similarity and margin thresholds, and median backend processing time. Results are shown as preliminary until at least 15 paired labelled decisions have been collected. False identifications retain a 5x penalty relative to returning Unknown.

The same data remains available through the admin WebSocket command `speaker_recognition/shadow_comparison`. Raw score values are deliberately never compared using a shared threshold because the two embedding spaces have different score distributions.

This mode is intended for evaluation only. A later change can expose an engine selector if real Home Assistant hardware results justify making ECAPA authoritative.


## Evaluation hygiene for future model changes

For offline comparisons, use consented real speech from both enrolled and
unenrolled speakers, recorded on the microphones and in the rooms where Assist
is normally used. Generated speech verifies integration behaviour, but cannot
establish real-person recognition accuracy. Keep evaluation audio outside the
repository unless appropriate consent, retention and redistribution rights exist.

The offline `speaker_recognition.evaluation` helpers expose separately:
- known-speaker identification and wrong-speaker rates (denominator: known trials);
- unknown-speaker correct rejection and false-accept rates (denominator: unknown trials);
- overall accuracy, error counts and median processing latency.

Never report a zero error rate when the relevant class has no trials: inspect
`known_trials` and `unknown_trials` alongside the reported rate.

Use `split_evaluation_records` with stable utterance-level `trial_id` values
**before** selecting thresholds. Paired Resemblyzer and ECAPA results with the
same trial ID are assigned to the same train/holdout partition, even if input
record order changes. Tune each engine's operating point on its training set,
then use only the untouched held-out examples for the comparison. Do not tune
again against the held-out results. A small holdout is diagnostic, not proof
of future accuracy; report its sample counts and data limitations.

This tooling does not change runtime recognition or automatically apply any
model or threshold change.
