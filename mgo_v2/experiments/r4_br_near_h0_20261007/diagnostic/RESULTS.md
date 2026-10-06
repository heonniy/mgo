# B64/input512: BR versus Near decode attribution

Completed one diagnostic per policy on physical GPUs0,1,4,5, R4/C30,
local B64, input512, decode32. Runtime is H0/full-pinned V3 P2/T2, BF16,
with the original frozen routing, teacher inputs and seed28. No added phase
barriers, new traces, policy tuning or extra workload cells.

**Observed improvement is lower expert-load imbalance and shorter collective
completion spans. There is no observed exposed H2D-wait saving. The original
4.32% primary gain is not reproduced in magnitude by this single diagnostic
pair, so its exact causal decomposition remains unresolved.**

## Timings kept separate

| ms/token | BR | Near | Near reduction |
|---|---:|---:|---:|
| Previous clean primary TPOT | 978.999 | 936.685 | 4.322% |
| Instrumented diagnostic TPOT | 1005.474 | 1000.410 | 0.504% |

Each global TPOT is the maximum measured rank duration. Diagnostic versus
previous primary TPOT increased2.70% for BR and6.80% for Near. These differences
combine instrumentation effects and run-to-run variation; this packet does
not isolate them. Do not scale diagnostic spans to the primary timings, claim
that the primary43ms gain is entirely explained, or discard either result.
Warm correctness passes are retained and are not primary timing repeats.

## Component comparison

These are **four-rank mean per-token service spans**, not a sum of independently
additive critical-path costs. Units are ms/token unless indicated otherwise.

| Component | BR | Near | Interpretation |
|---|---:|---:|---|
| Expert execution, excluding explicit H2D wait | 529.733 | 527.582 | Only0.406% less mean service time |
| Payload communication completion, forward+return | 114.050 | 105.726 | 7.299% less; includes peer arrival waits |
| Exposed H2D dependency wait | 0.000 | 0.000 | No wait_for_slot calls in either decode |
| H2D copy-stream service, overlapping other work | 205.787 | 207.214 | No service-time improvement observed |
| Decode expert copies/rank/token | 885.34375 | 885.30469 | Practically unchanged |
| Current-placement controller CPU | 18.390 | 19.643 | Near costs1.253ms more |
| Prefetch controller CPU | 32.818 | 32.925 | Essentially unchanged |

GPU event spans include host dispatch gaps, allocations and event overhead;
expert execution is not pure GEMM kernel-active time. Payload communication
excludes metadata and packet packing, and includes waiting for other ranks.
The H2D copy spans include the copy submission interval; they are not a
standalone calibrated PCIe bandwidth benchmark. Zero explicit H2D wait means
copies were ready at ready-first scheduling checks, not that copies were free
or had zero possible indirect interference with other GPU work.

## Load balance and communication

For each decode layer, take the maximum expert-token rows across ranks, then
sum those maxima and divide by32 decode steps:

- BR31,892.688 rows/token; Near26,106.406: **18.143% lower peak-rank workload**.
- Mean work is exactly24,576 rows/rank/token for both policies: the same total
  expert work is distributed more evenly, rather than removed.
- Expert service imbalance, sum over layers of(max rank time - mean rank time)
  divided by32, falls from62.405 to46.803ms/token: **25.001% smaller**.
- Sum of per-layer maximum expert-service durations falls592.139→574.385ms/token.
  This is an imbalance indicator, not a separately additive TPOT saving.
- Aggregate remote forward+return payload falls273.520→271.682MB/token,
  only**0.672%**. A7.299% reduction in collective completion spans therefore
  cannot be described simply as7.299% less network traffic.

These observations are consistent with improved rank balance reducing return
collective waiting. They do not isolate pure network-transfer time from peer
arrival or prove that load balance caused all of the timing difference.
In particular, the H0 expert service includes substantial host-side dispatch;
row balance alone does not translate proportionally into TPOT.

## Additive partition of the diagnostic critical rank

Rank0 has the largest total decode duration for both policies. Unlike the
mean-service table above, this current-stream partition plus outside-MoE
residual reconciles to that rank's diagnostic TPOT. Positive delta means Near
saved time. H2D DMA and the imbalance indicator must not be added again.

| ms/token, rank0 | BR | Near | BR minus Near |
|---|---:|---:|---:|
| Expert execution | 505.780 | 522.286 | -16.505 |
| Payload communication completion | 130.737 | 107.117 | +23.620 |
| Metadata service | 59.013 | 50.061 | +8.952 |
| Exposed H2D wait | 0.000 | 0.000 | 0.000 |
| Other MoE planning/packing/combining/host gaps | 275.823 | 278.600 | -2.776 |
| Outside MoE, including dense work and step overhead | 34.121 | 42.347 | -8.227 |
| **Diagnostic TPOT** | **1005.474** | **1000.410** | **+5.064** |

Rank0 executes slightly more service time under Near even though rank-average
expert time and per-layer imbalance improve. This is compatible with work
redistribution. It also demonstrates why independently summing maximum-rank
H2D, compute and communication measurements would be misleading.

## Audit and reproducibility

- All8 diagnostic rank receipts pass independent CPU state, slot-role,
  controller-count, copy+cancellation-bound and transport-count checks.
- All logits finite; no new compilation during either diagnostic.
- Each policy's output tokens match its warm pass and its original primary
  run exactly. BR versus Near differ at81/8448 output-token positions; frozen
  teacher inputs/routes remain shared. This packet does not assign a cause to
  those cross-policy BF16 output differences or claim policy bitwise parity.
- All12,288 MoE decode layer/rank/policy records retained, with metadata,
  forward, compute and return events, CPU controller times and copy traces.
- Per-rank partition checks reconcile to TPOT within0.01ms/token.
- 384/96GiB host memory guards retained; no OOM or failure. Worker+cleanup
  elapsed371s. Owned model-forward load restored on0,1,4,5 only; these are idle
  model loads, not continuing experiments. GPUs2,3,6,7 were untouched.
- Measurement implementation commit: e4bf862d8e03eebb3002b56585843cdade33f7c4.

Artifacts: [SUMMARY.json](SUMMARY.json), [RANK_COMPONENTS.csv](RANK_COMPONENTS.csv),
[LAYER_COMPONENTS.csv](LAYER_COMPONENTS.csv), [SOURCE_HASHES.json](SOURCE_HASHES.json),
and compressed per-rank traces plus validation receipts in[raw/](raw/).
