# Current execution and synchronization

Applies to main-table H0/full-pinned with optimized prefill and original decode.
Policy BR is selected for this audit; main-table OURS uses LA_CA_NEAR.
Full pinned storage is CPU RAM (54GiB/rank), not GPU cache. R4 C30 reserves
461/461/461/460 physical expert slots including two prefetch slots per rank.

## Prefill (48 layers)
1. Gather routing metadata; exact W128 probability tail; CPU placement and
   compiled prefill packet layout.
2. Async fused forward all-to-all; finish/unpack; CUDA event host wait ensures
   dispatch payload/unpack is physically complete.
3. Enqueue demand H2D from pinned CPU sources. Wait required slots on host.
4. Explicit distributed barrier plus current-stream synchronization.
5. H0 expert kernels; weighted BF16 contributions.
6. Per-destination-rank partial accumulation, return all-to-all, source combine.
   No separate explicit post-expert global barrier in this selected runtime.

## Decode (63 forwards, each 48 layers)
1. Live routing all-gather and D2H, CPU placement and original Python layout.
2. Enqueue demand H2D on dedicated copy stream, then async forward all-to-all.
3. Finish/unpack and local CUDA event host wait (T2); plan next-layer prefetch.
4. No global H2D barrier. Compute ready experts first; if none ready, wait for
   the selected expert's copy submission and insert a CUDA stream event wait.
5. No explicit post-expert global barrier. Local partial accumulation followed
   by return all-to-all; source ranks combine the returned partials.
6. Next-layer routing collectives couple ranks again. At each generated token,
   harness synchronizes the local GPU and records completion. TTFT/E2E use the
   latest rank timestamp from a common release; TPOT=(E2E-TTFT)/63.

All-to-all is a collective dependency, not an extra dist.barrier. An early
rank can wait for slower peers inside return/NCCL completion. The return span
includes upstream rank skew; do not equate it with pure wire transfer time.
Strict H2D-first serialized phases and post-expert isolation are OFF.

## Interpretation limits / implementation costs to inspect
- T2 has an explicit local host synchronization per layer; it constrains
  overlap but is part of the current chosen prefetch protocol, not a new probe.
- Decode still builds Python lists/indices, performs selected-ID D2H and scans
  slots. The owner restored this original path; this audit does not replace it.
- Per-layer unique-row layout validation remains enabled in MEASURE as a
  correctness guard. It uses Python sets and loops and is not a detailed profiler.
- Ready-first expert traversal also uses host lists and per-slot readiness polls.
- Diagnostic current-stream spans include host submission gaps, H2D dependency
  waits and peer skew. H2D service on the copy stream overlaps those spans.
- Cache hit denotes avoidance of a new demand copy. Promoted prefetch can be
  logically resident while its physical copy is still in flight.
