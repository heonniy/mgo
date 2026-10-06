# Owner amendment: joint TTFT, TPOT and E2E final validation

Owner chat authorizes32 decode steps in every setting's final physical
validation. Existing64 S1 prefill selection pairs finish unchanged. Select
one winning triple per eight R/B/L cells using S1 TTFT only. Replace the old
TTFT-only S2 queue at the completed-S1 boundary; no active measurement is
interrupted and no source used by S1 is modified.

Final protocol:
- Fresh process per selected cell; BR, LA_CA_NEAR and plain LA in that process.
- Same policy across prefill and decode; retain that policy's prefill cache.
- Untimed BR pass uses the existing frozen prefill and captures native decode
  routes, BF16 weights, gate history and autoregressive token inputs once.
-32 decode forward passes follow the first output from prefill (33 outputs in
  total). Fixed length, including if EOS is predicted; no early stopping.
- CPU replay builds physical copy/state proofs for all three policies from
  identical frozen routes/tokens. GPU warm passes must match proofs and all33
  output token IDs across policies before primary measurement.
- Keep strict serialized required H2D -> global barrier -> forward -> compute
  -> global barrier -> return for every prefill/decode layer. No prefetch,
  substitution or replication; decode uses the same unfused transport as
  prefill for this extension, avoiding another transport-policy change.
- TTFT: wall time through first-token GPU readiness. TPOT: wall decode duration
  divided by32. E2E: prefill start through final decode output readiness.
  Per-step durations and per-rank values remain available. Aggregate each
  metric by maximum across ranks. Capture/model load/CPU replay/compilation
  are outside primary timing. Teacher tokens/routes are staged beforehand.
- Two repeats for all three policies, reversed execution order. Evaluate the
  maximum first-two relative difference over TTFT/TPOT/E2E and all policies:
  <=2% stop;2-5% one third for all;>5% no third and instability flags. Retain
  every valid sample, ranges, mean for two / median for three.
- Owner follow-up removes all24 separate diagnostic passes and phase-time
  attribution. Keep only correctness/warmup and the bounded primary repeats.
  Result reporting reads physical-copy validation from existing measurement
  receipts and records TTFT/TPOT/E2E, ranges and instability. Strict execution
  barriers are retained; only extra instrumented runs are removed.

Interpretation: TPOT is measured after policy-specific prefill state and is
not an isolated decode-policy effect. A change in decode physical H2D count
is a placement-state effect. Seeds were optimized for prefill TTFT, not TPOT.
This remains BR-adversarial headroom rather than average-case performance.
A shared BR baseline is used for both final candidate comparisons.

Implementation uses `strict_decode32_worker.py`,
`run_strict_decode32_final.py`, and `report_strict_decode32.py`.
The original supervisor records a cancellation when the completed S1 queue
is handed off; `decode32_handoff.json` and `decode32_status.json` record the
intentional protocol replacement. Do not treat that cancellation as a GPU
correctness/OOM failure. No old TTFT-only S2 measurement should run.
