# Execution binding

Plan: `1a98d10ac17557e7a9112e12ad36d07fe5d5be27`.
Validated runtime: `40a63d8ba22d16684a358d4546bdae4783931e5e`.
Raw root: `/home/hwlee/mgo-results/local_remote_e2e_impact_20261001`.

The owner explicitly requests **R8 before R4**. Execute Stage A/R8, Stage B/R8
(local B8 then B4, followed by the user-requested B16 then B32 extension),
Stage A/R4, Stage B/R4. Only after all primary timings,
collect exactly two full-model Nsight repeats: R8/B8 C0 and C2, repeat 0.
The representative profile cell is fixed here before timing outcomes exist.
Optional 128-step confirmation is omitted to keep this bounded primary study
and its two required profiles; no timing-based policy tuning is allowed.

## Frozen execution

- Qwen3-30B-A3B-Instruct-2507 BF16, physical 9 MiB experts, H100 NVSwitch.
- Same GPU checkpoint, immutable CPU expert store, held-out-from-calibration
  GSM8K-train questions and finite similarity/affinity files as prior validation.
- Same-layer support64, **alpha=.25**, path support64, **eta=.5**, seed42.
- Coverage W128/k1/lambda2, gate .20/similarity .65, cache30, native arithmetic,
  direct slot views, one controller, no migration or replication.
- All GPU jobs sequential; NCCL warmup before pinned host-store loading.
- Reuse the prior R1/R4/R8 native parity and slot-view trace gates. The only
  runtime changes add locality/load counters to the existing metrics loop;
  no routing, admission, eviction, compute, or collective code changes.
  Stage A independently recomputes those counters on real captured events.

## Stage A

Capture one prefill plus eight decode events per layer, with exact routing
(substitution disabled), local B8. Sort the 384 decode events by active-expert
count, then capture index; select indices 0, floor((N-1)/2), N-1. Store real
hidden states, selected experts, weights, probabilities and reference outputs.

Each map uses 20,000 deterministic quota-preserving proposed swaps, seed20261001.
Endpoints minimize/maximize exact remote token-rank pair count; intermediate
targets interpolate those endpoint counts. These are heuristic endpoints,
not claims of proven global optima or exact locality percentages. The same
initial assignment, proposals, quota and solver family apply to all maps.

Load a fresh map through GlobalExpertController's fixed admission adapter
outside timing. No map migration occurs inside a run. Verify every output
bitwise against the captured output. Time dispatch/expert execution/combine
with the controller's fixed hit-only plan (routing metadata/solver excluded).
Three warmups, then 20 uninstrumented iterations and 20 separate CUDA-event
diagnostic iterations per map. Both verify zero physical slot fetches and
unchanged slot residency. The previously validated unchanged direct-slot binary
supplies the independent evidence against hidden per-hit weight uploads/D2D.
Report max-rank wall latency and max-rank summed dispatch/combine CUDA intervals;
the latter include stream/launch waits and are not isolated NCCL kernel time.

## Stage B

Five repeats for each of fifteen cell/policy combinations, no profiler/CUDA-event
instrumentation. Two warmup forward calls per worker job, then fresh logical
and physical cache/history before each measured repeat. Dense weights and
allocator/kernel caches stay loaded; expert cache persists within a repeat.

**64 decode steps means prefill + 64 decode forwards (65 generated tokens).**
TTFT is the first forward plus greedy selection; TPOT is the mean of the 64
max-rank decode step times. Whole-generation time is max-rank continuous wall
time, including normal loop overhead. EOS ends answer scoring only; all rows
continue fixed work, so this is not serving throughput or a long-horizon quality
evaluation. Save every token, including fixed work after EOS, to audit repeat
determinism. Publish prefill/decode/full-generation counters separately.

Submitted peer payload is independently derived from the unchanged packet
protocol: dispatch = remote pairs × (2048×2 + 8 + 8×(8+2)); native-order return =
remote effective expert routes × (2048×2 + 16). This excludes self traffic,
router/count all-gather metadata and NCCL wire/protocol overhead. Diagnostic
Stage-A CollectiveStats verifies this formula against actual submitted tensors.
Supplementary `all_submitted_peer_tx_bytes` also includes routing/count
all-gathers, derived from actual valid prompt lengths, model tensor dtypes,
world size and decode length, and checked against posthoc CollectiveStats.
Stage-A NCCL diagnostic intervals include the count exchanges as well as
dispatch/return payload calls. None of these are wire-byte measurements.
Logical H2D bytes are physical dispatcher fetches × 9 MiB; only posthoc traces
are labeled actual observed H2D bytes. Controller time is per rank; report its
max plus full per-rank evidence. Event CV is the unweighted event mean.

## Gates and reporting

CPU policy, mask and locality-map tests pass before launch. Audit every rank,
repeat, token, source/input hash, physical fetch count, frozen configuration,
and completion receipt. Runtime hashes are stored before launch and checked
against per-worker provenance. Report all repeats and median/min/max; publish
negative outcomes. Profiled times are diagnostic only. Stop for owner review
after the required reports and two profiles, without automatic retuning.

## User-requested expansion during execution

The owner explicitly requested R8 local B16 and B32 in addition to the original
B8 and B4 cells. Each additional batch uses all three frozen policies, five
repeats and 64 decode steps: 30 additional generations, 75 total. The original
R8/B8 and R4/B8 acceptance anchors and the two preselected R8/B8 profiles remain
unchanged. B16 and B32 use the first 128 and 256 questions from the same bound
workload; no new calibration, coefficients or policy search is introduced.

Only the scheduling parent is paused while the existing R8 workers finish.
The extension waits for successful torchrun termination and empty GPUs, runs
the new R8 cells, then resumes scheduling of R4. The original R8 launcher's
outer wait includes this scheduling pause; individual generation timers do not.
`scheduling_pause.json` and `extension_status.json` preserve this transition.
An independent fresh worker job supplies the same two-call warmup for the
extension. Runtime source and compiled-binary fingerprints stay unchanged.

W128 is a window of global routed-token rows, not 128 decode steps. In the
rank-major gather order, R8/B16 contributes all 128 current decode rows, while
R8/B32 contributes 256 and the history retains the last 128 (ranks 4–7).
All 256 tokens still execute. This existing frozen history behavior is retained
and disclosed; no batch-dependent history retuning or row reordering is made.

For a fresh reproduction, `scripts/run_local_remote_study.py` now includes
R8/B16 and B32 by default and schedules them before R4 directly, without a
scheduling pause. `--dry-run` writes all three cell manifests and the job order
without creating a CUDA context. Use a fresh output directory for new runs.
