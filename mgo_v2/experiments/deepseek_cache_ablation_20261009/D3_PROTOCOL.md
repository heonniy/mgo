# D3 sampled layer critical-path probe

Use the D1 fixed-continuation and frozen-router files unchanged. Run one
guarded C20 and one guarded C50 target with
`MGO_DEEPSEEK_CACHE_DIAG=1` and `MGO_DEEPSEEK_DIAG_DECODE_STEPS=8`. All 64
generation steps still execute; detailed collection covers the first eight
decode steps (26 routed layers each), after the prefill token. The target
remains a diagnostic and must not replace D1 clean TPOT.

For each `(step, layer, rank)` record:

- Metadata host pack, all-gather submission, receive-to-host wait and parse/
  gate-history time; the receive-to-host wait can include collective completion.
- Controller, rank-partial layout/slot binding, H2D enqueue and dense-payload
  submission host spans.
- Dispatch, expert and return-plus-combine CUDA-stream intervals; the
  dispatch-to-combine interval; explicit H2D wait, expert groups/rows/waves,
  local fetches, peer bytes, `ready_many` host time and native-wave host time.
- Host monotonic dispatch and return submission timestamps across ranks.
  Their rank spread is **submission skew**, not a direct NCCL arrival time.

CUDA events are read after the existing per-token synchronization; no new
per-layer synchronization or global barrier is inserted. CUDA intervals can
include host launch gaps and collective peer waits, and the phases may
overlap H2D, so do not add them into a synthetic TPOT or call a collective
interval wire time. Compare diagnostic TPOT with D1 clean samples to assess
profiling overhead, and preserve all raw layer records outside Git. Summaries
must use the actual max-rank per layer and show rank/step variability.
