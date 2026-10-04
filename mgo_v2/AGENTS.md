**OWNER OVERRIDE (2026-10-04): transport follow-up is c60, substitution OFF.**
For `experiments/transport_stack_remeasure_20261004`, ignore the older cache30/sub-ON wording below. Run R=MATH/R4/local-B64/Gate/decode256 on GPUs 0,1,4,5 with cache=60% and substitution OFF. Use CoSLoT-style pinned H2D for all timed cells and compare current/coslot/coslot-active for BR/CA. PLAN identity is keyed by c60+s0; create and validate missing c60/s0 reference PLANs outside timing rather than requesting a path from the owner. Execute `scripts/run_transport_stack_packet.py`. Env1 first, Env2 only if all six Env1 conditions meet <=5% E2E and TPOT spread. Commit each checkpoint and stop after the packet.

**OWNER RESTART (2026-10-04): transport stack at ac6d6bc.**
Follow `experiments/transport_stack_remeasure_20261004/{README.md,PLAN.md,EXECUTION.md}`.
The missing-PLAN block is superseded: create current/pageable R/BR and R/CA
reference PLANs if absent, validate once, freeze, and reuse across all cells.
Run R4 GPUs 0,1,4,5, pinned H2D with two buffers, BR/CA, current/coslot/
coslot-active, three MEASURE samples per condition. Env1 first; Env2 only if
all E2E and TPOT ranges/medians are <=5%. No additional repetitions.
OldCA remains cancelled. Commit each phase and restore owned model workers
on completion/failure. Do not resume old experiment queues.

**OWNER QUEUED FOLLOW-UP (2026-10-04): OLD-CA token->rank fan-out isolation.**
Do not interrupt the active `fetch_matched_b32_a2a_20261004/tolerance_001`
study. After it fully exits and commits, run
`experiments/old_ca_fanout_followup_20261004/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Reuse ShareGPT B32/decode64/cache30/Gate W128/substitution-OFF frozen traces.
Compare BR, Current-CA, and an `OldCA-fanout` policy porting the exact
a0be82e `_incremental_cost` balanced-Hungarian objective: minimize incremental
token->remote-rank destinations created by incoming experts relative to
pre-owned destinations. No same/path affinity in this first isolation. Prefer
a common <=0.1% BR-relative fetch/H2D-matched workload; bounded fallback <=0.25%
without padding or fake fetches. Env2 only, A3 only, R4(0,1,4,6) then R8(0..7),
single-shot BR/Current-CA/OldCA first; if a CA variant gains >=1% TPOT/decode
wall, allow exactly one additional BR+that-policy confirmation pair. Commit
results and stop; no automatic A2/affinity/decode256 expansion.

**OWNER FETCH TOLERANCE AMENDMENT:** Resume the B32/decode64 frozen-route pivot with abs(CA-BR)/BR <=0.001 for both total fetches and H2D. Reuse completed CPU receipts, select Peer-best and Critical-best independently, deduplicate identical winners, then Env2 R4(0,1,4,6) -> R8(0..7), A3/A2 single-shot; exactly one confirmation pair only when any initial CA gain >=1%. No new trace capture or expanded search. See `experiments/fetch_matched_b32_a2a_20261004/tolerance_001/EXECUTION.md`.

**OWNER IMMEDIATE REPLACEMENT:** The owner cancelled all remaining old R4 physical experiments and their queue. Execute commit81dfb0c `fetch_matched_b32_a2a_20261004` now after stopping/reaping those processes. Reuse only the first64 decode steps of the existing exact256 trace; no new trace capture. This overrides the older wait-for-R4 wording below.

**OWNER QUEUED FOLLOW-UP (2026-10-04): FAST fetch-matched B32 + 2-A2A pivot screen.**
Do not interrupt the active `ca_stress_physical_validation_20261004` R4 work.
After it fully exits and commits, run
`experiments/fetch_matched_b32_a2a_20261004/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Use ShareGPT local-B32/cache30/Gate W128/substitution-OFF/**decode64**, reusing
the first 64-step prefix of the existing exact decode256 traces (no recapture).
Search two independent fetch/H2D-matched winners per R: Peer-best (aggregate
peer-byte reduction) and Critical-best (sum of per-event max-rank send+recv
reduction); deduplicate identical winners. Physical replay uses common frozen
routes/tokens in Env2, R4 first on GPUs0,1,4,6 then R8 on GPUs0-7, comparing
current A3 (3 A2A) against A2 (routing-weight A2A removed). Fast screen is
**one timed run per BR/CA/A3/A2 condition**, with no full-horizon warmup per
MEASURE; only one-time compile validation and short 8-16-step readiness warmup
per (R,runtime). If a BR-vs-CA pair shows >=1% positive CA gain, run **one
additional confirmation pair only** (BR once + CA once), then stop repeating.
Commit and stop; no decode256 extension or other automatic follow-up.

**OWNER TIMING REPETITION AMENDMENT:** Active/queued physical R8 and both R4 GPU-set studies use two clean MEASURE repeats per cell/environment/policy. Gate E2E and TPOT with abs(T1-T2)/mean(T1,T2): both <=2% stop at two; maximum >2% and <=5% adds exactly one third; either >5% stops at two and is unstable. Never use one sample as primary evidence or add further repeats. This overrides earlier three-plus-two repetition text. See `experiments/ca_stress_physical_validation_20261004/REPETITION_AMENDMENT.md`.

**OWNER EXTENSION (2026-10-04): add R4-best physical stress cell after current R8 run.**
Do not interrupt or mutate the in-flight R8-best validation. After it fully
exits and commits, run `experiments/ca_stress_physical_validation_20261004/R4_AMENDMENT.md`.
Use ShareGPT R4/local-B8/cache30, Gate W128, substitution OFF, decode256,
sample205/dp4/BR-seed73 on physical GPUs 0,1,2,3. Frozen CPU expectation is
~30.1124% chosen-seed BR->CA peer reduction (eight-seed range
29.6355%--30.1124%). Compare BR vs CA in Env1 and Env2 with three clean repeats
using the validated BOUNDARY + fixed-affinity harness. Controller/Hungarian
work remains outside MEASURE. Commit R4 results and stop; no further cell is
authorized.

**OWNER PRIORITY (2026-10-04): one physical CA stress validation.**
The timing harness is HARNESS_STABLE and the CA stress search is complete.
Run exactly `experiments/ca_stress_physical_validation_20261004/` next.
Use ShareGPT R8/local-B8/cache30, Gate W128, substitution OFF, decode256,
sample_seed81, dp_seed86, BR seed42. Compare BR vs CA physically in Env1 and
Env2 with three clean repeats each using the validated BOUNDARY + fixed-affinity
harness. Hungarian/controller work remains PLAN-only, outside MEASURE.
This is an optimized communication-stress workload, not dataset-average.
Commit results and stop; do not expand the matrix automatically.

**OWNER QUEUED FOLLOW-UP (2026-10-04): CA-favorable sample/DP search after timing diagnosis.**
Do not interrupt the active `timing_stability_numa_20261004` packet. When that
diagnostic finishes, regardless of PASS/FAIL, proceed directly to
`experiments/ca_stress_workload_search_20261004/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
This follow-up is resource accounting, not E2E timing. Freeze Gate W128,
decode256 and substitution OFF. BR and CA must retain the identical per-event
balanced mandatory-miss quota (rank max-min <=1). Reuse the existing 512 exact
traces and, using all 8 H100s, extend each dataset candidate pool to 2048
requests when the current filter permits. Then search sample selection and
equal-size DP rank assignment on CPU; exact-replay the top candidates across
R={4,8}, local B={8,16,32,64}, cache={30,40,50,60}. Select structural
CA-favorable sample/DP pairs using median savings across fixed BR seeds, and
also report the best observed BR seed separately. Commit results and stop;
do not auto-launch physical timing.

**OWNER AMENDMENT (2026-10-04): broaden timing-stability diagnosis; guest NUMA is not host NUMA.**
S0 shows a KVM guest with one visible NUMA node and no exposed GPU PCI NUMA ID.
Do not infer that the physical host has one NUMA domain, and do not attempt to
invent same/cross-NUMA labels. Finish the already-started S1 HEAVY/BOUNDARY
sequence unchanged, then test: (H2) pageable file-backed expert page residency
via NORMAL/PRETOUCH, (H4) 1/2/4/8-GPU concurrent 9-MiB H2D scaling, and (H5)
an empirical Env2 SHM GPU-pair matrix at 32/128/512 KiB. Use fixed disjoint CPU
affinity for S3. See `experiments/timing_stability_numa_20261004/HYPOTHESES.md`
and the appended PLAN amendment. No policy timing resumes before <=5% stability.

**OWNER PRIORITY (2026-10-04): PAUSE policy timing; validate measurement stability first.**
The physical BR/CA/CA-rep E2E matrix is suspended. Preserve completed receipts
but launch no new policy-comparison cell after any already-running bounded run
finishes. Read `experiments/timing_stability_numa_20261004/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Use the exact noisy P/CA-rep workload, first as a decode64 prefix, to isolate
the current heavy PSS monitor versus a boundary-only no-concurrent-monitor
timed region. Then characterize 9-MiB local/remote NUMA H2D and Env2
same-NUMA vs cross-NUMA SHM at only 32/128/512 KiB. Freeze rank CPU/NUMA
affinity and require <=5% within-environment timing spread before any policy
timing resumes. If stable, confirm with three decode256 repeats; otherwise
stop and report instability. No BR-vs-CA, CA-vs-CA-rep, seed/sample timing, or
broader sweep is authorized until HARNESS_STABLE. Do not automatically resume
the old matrix.

**NEW OWNER PRIORITY (2026-10-03): physical Env 1 / Env 2 E2E + TPOT offloading.**
Read `experiments/env_e2e_tpot_offload_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
This is a real Qwen3-30B-A3B expert-offloading experiment, not a byte/NCCL-only
replay: CPU-resident experts must undergo real miss H2D into bounded GPU expert
caches and routed activations/experts must execute physically. The bounded
matrix has P=ShareGPT256 R8/B16 cache40 Gate sub-ON, R=MATH256 R4/B64 cache30
Gate sub-ON, and E=P with sub-OFF BR/CA control; policies are BR/CA/CA-rep as
listed. Env 1 is validated P2P/IPC; Env 2 is P2P-disabled validated SHM.
Crucially, every cell is split into PLAN -> COMPILE -> MEASURE -> COUNTERS.
Controller/oracle work, compilation, profiling, per-event logs and hit/miss
instrumentation are forbidden inside MEASURE. COUNTERS is a separate untimed
pass. Primary timing is E2E and TPOT with three counterbalanced clean repeats;
only the frozen noise gate can add two targeted repeats. Gate W128 is the
physical eviction policy; replicas are not protected. Do not expand LRU/cache/
batch/R axes or tune thresholds. Stop resident load workers before science and
restore them only after all experiment processes exit.

**Latest owner resource policy (2026-10-03, after packet completion).**
For future authorized CPU experiments, maximize useful throughput using available
CPU affinity and measured memory headroom. Benchmark a small initial wave, then
increase concurrent independent cells while throughput improves; avoid nested
BLAS/OpenMP oversubscription, redundant runs, and repeated trace loading. Reuse
validated completed cells and shared/read-only inputs. Reserve host memory for
other users and model workers; enforce per-worker and aggregate memory limits
and reduce concurrency under contention. The old 16-worker limit belongs to the
completed packet, not a universal limit for future experiments. Record the
chosen concurrency, measured throughput and peak RSS with each future run.
No new scientific matrix or rerun is authorized by this resource preference.
Dynamic refresh remains stopped. After GPU experiments, the owner now explicitly
requests resident model inference load on all GPUs 0--7, superseding the older
four-GPU restriction. Use `examples/model_inference_load.py` (neutral process
name), currently batch 1024. Preserve its memory/temperature/foreign-process
guards and hand GPUs back before an authorized experiment starts. See
`operations/model_inference_load_20261003/` for the restart verification.

**BR / CA / CA-rep two-horizon packet COMPLETE (2026-10-03).**
Read `experiments/br_ca_carep_cpu_headroom_20261003/{RESULTS.md,validation.json}`.
Two exact 512-request 256-decode master traces completed on all eight GPUs
in one loading session; no recalibration was needed. Both owner-selected CPU
horizons completed: 1536 main + 16 BR seed-audit cells. BR/CA match exactly
across 512 paired 64-step prefixes; 168 zero-replica cases match CA state.
CA_HEADROOM 283/512, CA_STRONG_HEADROOM 33/512. CA median peer reduction
is 9.32--13.07% across dataset/horizon groups; CA-rep median is 0--3.01%,
maximum 17.67%, so neither replica 20% label passes. No quality or timing claim.
Replay aggregate RSS peaked at 6.64 GiB; all capture workers exited and GPU
processes were empty at completion. Dynamic refresh remains owner-stopped at
99/120. Stop for owner review; earlier instructions below are history.

**OWNER CPU HORIZON AMENDMENT (2026-10-03): run both 64 and 256.**
The owner explicitly selected both horizons after 986ba64: 1536 main CPU
replays plus at most 16 seed-audit cells total. Read the current packet's
EXECUTION.md. GPU 0--7 use is explicitly authorized; previous resident load
workers were stopped. Dynamic refresh was stopped by owner at 99/120 cells
and must not resume automatically. Earlier scope statements below are history.

**OWNER TRACE AMENDMENT (2026-10-03): decode256 master captures.**
For the active BR / CA / CA-rep packet, capture MATH and ShareGPT once each on
8 GPUs with exactly 256 decode tokens. decode64 is the first 64-step prefix of
the same trace; no separate 64-token GPU run. Add only the lightweight raw
routing 64-vs-256 horizon audit from PLAN.md. Do not double the CPU matrix
without a later owner instruction. The FineWeb-Edu SERE calibration and all
other BR / CA / CA-rep axes remain unchanged.

**REVISED OWNER PRIORITY (2026-10-03): BR / CA / CA-rep CPU headroom with SERE calibration + two workloads.**
Read `experiments/br_ca_carep_cpu_headroom_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Substitution calibration is NOT MATH/ShareGPT-specific. First reuse the existing
similarity artifact only if provenance proves the same Qwen checkpoint,
FineWeb-Edu 400 sequences x128 tokens, and SERE Frobenius output similarity;
otherwise calibrate exactly once with that SERE-style setting. Workloads are
(1) MATH test 512 and (2) ShareGPT V3 cleaned 512 conversational turns selected
for naturally long responses; both use fixed 64 generated tokens for controlled
decode-heavy traces. Load the 8-GPU model once, optionally calibrate, then
capture the two 512-request master traces sequentially. Derive all R={4,8},
B={8,16,32,64} workloads offline. Sweep global cache 30/40/50/60%, LRU/Gate,
substitution OFF/ON, and BR/CA/CA-rep. Report exact/local/substitute/effective
hits, residual miss, turnover/reload, H2D, peer and policy headroom. Maximum
768 main +16 seed-audit CPU replays, up to 16 single-thread cells concurrently.
No Env timing, accuracy, NCCL, B128, Coverage eviction or workload-specific
similarity calibration.

**Owner-authorized next study (2026-10-03): dynamic stale-replica refresh.**
Read `experiments/dynamic_replica_refresh_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
The completed d666414 sweep showed cache relief can make replicas survive while
peer traffic rises because the historical duplicate cap stops new admissions
once full. Reuse the validated B8/B32 traces and test duplicate-to-duplicate
refresh only: N0 no-refresh, C1/C2 current exact-peer refresh, and O4/OR future
route-demand oracles. Primary B32 uses cache40/60, LRU/GATE, substitution
OFF/ON, rho .125/.25; B8 cache60 is secondary. 120 CPU cells total. A refresh
may evict only an inactive duplicate, never the final global copy or an active
copy, and must preserve unique coverage at the swap instant. First reproduce
all N0 cells exactly. Report H2D/peer/reloads/locality, stale age/lifetime and
resource-price break-evens. No GPU/model, quality, E2E/NCCL timing, new capture,
B64/B128, R8 or retuning. Commit and stop for owner review.

**Cache/eviction/substitution packet complete (2026-10-03).**
Read `experiments/cache_eviction_substitution_20261003/{RESULTS.md,validation.json}`.
Two compact gate-history captures match historical tokens/routes/cache hashes.
All 264 CPU cells passed; the five historical B8 points and final states match
exactly, and six cache30 fixed-460 controls match their rho=.25 counterparts.
CACHE_RELIEF, EVICTION_HEADROOM, SUBSTITUTION_SYSTEM_HEADROOM and
REPLICATION_REGIME_SHIFT have predeclared witnesses. Fixed-460 B8/LRU cache30
versus cache60 cuts H2D 219.076->39.797 GiB but raises peer 185.297->328.332 MiB;
replica survival>=48 rises .87%->85.37%. B32/cache30/LRU/OFF lambda_first is
97.071643 versus the B8 420.223139 anchor. No B8 configuration meets the 4x
price-reduction clause (minimum 154.174); B8 regime witnesses use historical
point dominance instead. These are frozen-route byte results, not timing or
quality evidence. CPU replay peak RSS 893.18 MiB under hard 8-GiB address space.
GPU 0/1/4/5 model workers were restored; 2/3/6/7 were untouched. Stop for owner
review: no quality run, new model capture, timing, tuning or policy follows
automatically. The original authorized scope below is retained as history.

**Owner-authorized cache/eviction/substitution characterization (2026-10-03).**
Read `experiments/cache_eviction_substitution_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
The historical replication frontier was cache30 + LRU + exact-only. Test whether
that regime caused the negative result by sweeping cache={30,40,50,60}%,
eviction={LRU,GATE,COVERAGE}, substitution={OFF,ON}, and
rho={0,.125,.25,.5,.75}. Substitution is frozen at gate protect .20 and
similarity .65; coverage uses W=128/k=1/lambda=2. If needed, create exactly
two diagnostic B8/B32 exact captures that store compact per-event gate-history
score vectors and verify tokens/routes against existing captures. Historical
cache30/LRU/exact B8 rho coordinates must reproduce exactly before the sweep.
Run 120 B8 cells + 24 fixed-460-duplicate B8 controls + 120 B32 cells, CPU-only
after capture. Report H2D/peer frontiers, reloads, unique coverage, replica
lifetime/reuse and substitution system counters. No quality claim/run, B64,
B128, R8, NCCL timing or physical F/K/C is authorized. Stop for owner review.

**Owner-authorized mechanism-isolation study (2026-10-03): synthetic communication-price crossover.**
**Complete: TIMING_UNSTABLE; exact C0 RESOURCE_PRICE_SHIFT retained.**
Read the packet's RESULTS.md and validation.json. Exact lambda crossovers
are 420.223139, 562.335956, 905.702210 and 1339.761831. Original five-rho
totals/state hashes and all 20 T0 actual/self cells passed correctness,
but negative remote premiums and rho=.75 control disagreement fail C2.
No C3/C4 time-price model, extra timing, H2D or R3 run is accepted.
Stop this packet. Our GPU 0/1/4/5 model workers were restored; 2/3/6/7
and their jobs were untouched. The original authorized scope below is history.

Read `experiments/synthetic_comm_price_crossover_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Use only the existing B8 five-rho frontier. First commit the exact CPU
resource-price sweep, then freeze all five communication/fetch schedules and
run only T0 actual-vs-matched-self communication traces on GPU 0/1/4/5.
Reuse the existing 9-MiB H2D calibration. If the frozen timing gate passes,
sweep `J=H2D_model+T_self+gamma*T_remote` and solve policy crossovers.
R3 is historical unstable context only: no R3 rerun and no PCIe equivalence.
No model, R8, longer decode, batch/cache sweep, substitution, new replica
policy or physical F/K/C timing. Do not touch GPU 2/3/6/7.

**Hot-expert threshold packet complete (2026-10-03): CURRENT_BATCH_TOO_COLD under the pooled-median gate.**
Read `experiments/hot_expert_replication_threshold_20261003/{RESULTS.md,validation.json}`.
H0 and the bounded H1 calibration passed. Concurrent-H2D thresholds are
T0 >512 and R3 512; B32 max remote demand is 32, so H3 has no candidates
and no B64/B128 capture follows. R3 pass crossovers differ (512 versus 1),
and single-rank pooled p90 crosses at 1; no robust intrinsic crossover is
established. Preserve these sensitivity results, not just the gate label.
Stop research work. Only our model workers on GPU 0/1/4/5 were restored;
do not restart our owner-stopped workers on GPU 2/3/6/7 or touch new jobs
there. No extra calibration, model run, threshold tuning or controller is
authorized by this result. The original scope below is historical.

**Owner-authorized follow-up (2026-10-03): hot rank-local expert replication threshold.**
Do **not** preempt the active `future_rank_affinity_placement_20261003`
packet. After that study commits and stops, read
`experiments/hot_expert_replication_threshold_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
First characterize current `(layer,expert,rank)` token-demand hotness on the
existing B8/B16/B32 traces. Then run only the bounded model-free calibration
that compares one remote dispatch+combine pair for n={1..512} rows against one
9-MiB expert H2D under T0-IPC and R3-SHM (isolated and four-rank concurrent
H2D). Derive measured median/p90 demand crossovers without extrapolation and
map them back to trace coverage. Only measured thresholds may drive the
capacity-aware threshold replay. B64/B128 are not automatic; use the
predeclared H4 gate to decide whether B64 is worth proposing. R3 remains a
synthetic P2P-disabled SHM stress condition, not PCIe-only. No R8,
substitution, future predictor, cache sweep, Nsight, NCCL tuning, or physical
F/K/C model timing.

**Owner-authorized CPU-only follow-up (2026-10-03): future rank-affinity single-copy placement.**
**Completed: NO_PLACEMENT_HEADROOM.** All 18 CPU cells and 18 deterministic
verification invocations passed. B8 O0 peer reduction is 4.9645%, below the
frozen 5% modest gate; every OH policy has more peer bytes than O0. Peak RSS
707.75 MiB. Read the packet's RESULTS.md and validation.json. Do not rerun.
The owner explicitly requested the queued hot-expert threshold packet next.
Preserve the owner's shutdown of GPU 2/3/6/7; only workers on 0/1/4/5 may
be paused/restored for that new packet. Original scope below is historical.

The selective-replication study at `6a8127c` ended `NO_HEADROOM`, but
rank-local recurrence was high. Change the action, not the signal: on an
unavoidable global miss/reload, keep exactly one expert copy and choose which
currently-demanding rank owns it. Read
`experiments/future_rank_affinity_placement_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
First reproduce B8 F exactly, then compare F, current exact-byte oracle O0,
and OH1/OH2/OH4/OHremaining. Run the same six policies on existing B16/B32
as secondary traces: 18 CPU cells total. No duplicates, migration, prefetch,
substitution, model/GPU/NCCL, new capture, R8 or cache sweep. Separate F->O0
(current placement quality) from O0->OH (future affinity value), compare B8
against the committed old replication frontier, commit compact results, and
stop. Do not disturb the owner's eight resident-model GPU workers.

**Rank-local reuse study complete (2026-10-03): NO_HEADROOM.**
Read `experiments/rank_local_reuse_oracle_20261003/{RESULTS.md,validation.json}`.
B8 F was reproduced exactly. The four-step marginal-byte recurrence share
is 73.70%, but none of the 16 selective cells dominates an old nonzero-rho
point. B16/B32 secondary traces also completed 16 cells each; no historical
secondary K/C frontier is assumed. All 48 CPU cells passed, with peak RSS
720.78 MiB under a hard 4-GiB address-space bound. The original eight
resident-model GPU worker PIDs remained unchanged. Stop research work;
no online controller, new trace, threshold tuning or GPU follow-up is
authorized by this result. Keep the requested resident-model idle load.
The prospective instructions below are retained as history.

**Owner-authorized CPU-only headroom study (2026-10-03): rank-local reuse and selective replication.**
Read `experiments/rank_local_reuse_oracle_20261003/{README.md,PLAN.md,AGENT_TASK.md,matrix.json}`.
Reuse the existing exact B8 capture and the same rho=0 cache/traffic semantics.
First reproduce F exactly, then measure temporal reuse of remote
`(layer,expert,requesting-rank)` pairs and exact future marginal peer-byte
savings for H={1,2,4,remaining<=8}. Apply the predeclared reuse gate before
any selective replay. If it passes, run only 4 horizons x 4 fixed future-saving
thresholds with the real cache slots/LRU, compare the resulting H2D/peer
coordinates against committed F/K/C, and stop. B16/B32 are secondary existing
traces only. CPU only: no model/GPU/NCCL, no R8, no substitution, no new F/K/C
physical timing, and do not disturb the owner's resident-model GPU workers.

## New priority — Fetch/Communication Pareto with P2P-disabled H100 (2026-10-02)

**Batch communication packet complete (2026-10-03), including local B32.**
B4/B8/B16/B32 union message medians are 16/32/64/128 KiB; nonzero message
counts stay approximately 9,216. All three new source captures, two transport
smokes, 20 calibration cells and 16 trace cells passed. Whole-trace CUDA
R3/T0 median ratios are 1.1283/1.5915/0.9435/1.2345. B4->B16 is MIXED;
the B4->B32 endpoint rule is BATCH_SENSITIVE (+10.62 percentage points),
but the curve is nonmonotonic, B4 reverses direction between passes, and
calibration fits are weak. No robust bandwidth crossover is established.
CPU Pareto files are unchanged; no F/K/C or extra repetitions are authorized.
Read `BATCH_COMM_SENSITIVITY_RESULTS.md` and `batch_comm_validation.json`.
Stop research experiments; retain the owner's eight-GPU resident-model
inference load during idle time, subject to the documented resource guards.
Earlier experiment priorities below are historical.

**Owner amendment (2026-10-03): include local B32/global B128.**
The batch packet now has B4/B8/B16/B32, with exactly three new captures
(B4/B16/B32) and the existing B8 reused. This supersedes the plan's B16 cap.
The owner requests idle load on all eight GPUs whenever experiments are not
running, and then specified real model inference rather than GEMM burn.
Use `examples/model_inference_load.py`: resident Qwen1.5-MoE-A2.7B-Chat,
BF16, current idle local batch1024/max128 input tokens, repeated forwards
without KV growth; last-token logits only. This idle batch is separate from
the completed B4-B32 research matrix. Read `MODEL_INFERENCE_LOAD.md` in the
experiment packet for live settings and measured utilization.
Pause our eight inference workers for experimental stages; resume on exit,
including failure. Never kill another user's jobs. Workers yield to another
compute process, temperature >=85 C, GPU free <8 GiB or host free <128 GiB.
See `BATCH_COMM_EXECUTION.md` for the frozen expansion and measurement rules.

**Immediate follow-up after the B8 IPC/SHM characterization: batch communication sensitivity.**
Read `experiments/fetch_comm_pareto_p2p_20261002/BATCH_COMM_SENSITIVITY.md`.
Hold cache ratio at 30% and R4 fixed; compare local B4/B8/B16 (global
16/32/64). Reuse B8 and capture only exact-only B4/B16 traces with the same
prompt ordering, P0 seed42, LRU, no substitution/replication, 1 prefill +
8 decode. Replay all three 384-event traces on stable T0-IPC and R3-SHM.
Primary = whole-trace CUDA/wall ratio; old sum-event max-rank is diagnostic
only. Also report message-size/fan-out geometry and a bounded 16-256 KiB
small-payload calibration. No cache sweep, F/K/C model runs, R8, Nsight,
NVLink-mode work or NCCL tuning. Commit and stop.

**NVLink bandwidth ladder blocked (2026-10-03): BLOCKED_PRIVILEGE.**
All eight GPUs are idle, but the read-only bandwidth-mode query returned
code 4 (insufficient permission per the installed NVIDIA-SMI manual).
Current FULL state cannot be verified. No mode write, calibration, trace,
model or CPU screen was run. CPU results and frozen schedule metadata
remain SHA256-identical. Restoration is not applicable because no mode was
changed. Read `NVLINK_BW_LADDER_RESULTS.md` and
`nvlink_bw_ladder_validation.json` in the experiment packet.
Stop without sudo, mode writes or automatic retries. Earlier checkpoints
below are historical.

**Immediate follow-up after `3e59975`: NVLink bandwidth ladder only.**
Read `experiments/fetch_comm_pareto_p2p_20261002/NVLINK_BW_LADDER.md`.
Characterize the same frozen MoE communication trace under FULL-P2P,
NVLink-bandwidth-OFF direct P2P, and P2P-disabled SHM. Use
`NCCL_CUMEM_ENABLE=0` throughout. The bandwidth-mode write is treated as
server-global: proceed only if all 8 GPUs are idle and current mode is FULL;
restore FULL unconditionally after OFF and verify it before any further work.
Run only 32/128-KiB tiny calibration and the 384-event trace. No model, CPU
Pareto rerun, extra bandwidth modes, NCCL tuning, sudo, driver reset or job
killing. Commit and stop for owner review.

**IPC baseline rebase complete (2026-10-03): AMBIGUOUS_GAP.**
Both NCCL_CUMEM_ENABLE=0 smokes passed: T0 uses P2P/IPC only and R3
uses SHM/direct/direct only, on one node/four local ranks. The original
384-event communication replay passed all payload checks in both orders.
Primary R3/T0 ratios are 1.0529x and 1.0836x (median 1.0682x), below the
1.20x STRONG_GAP threshold. CPU Pareto results and frozen schedule metadata
are SHA256-identical; no CPU screen or model was run. Peak process-tree RSS
was 2.72 GiB with no OOM. GPUs 0,1,4,5 are released. Read
`IPC_BASELINE_REBASE_RESULTS.md` and `ipc_baseline_validation.json` in the
experiment packet. Stop here; do not launch clean F/K, add repetitions or
tune transport. Earlier checkpoints below are historical.

**Immediate follow-up after `f64eb56`: stable IPC baseline rebase.**
Read `experiments/fetch_comm_pareto_p2p_20261002/IPC_BASELINE_REBASE.md`.
Do not rerun the CPU Pareto screen; its raw H2D/peer-byte results and F/K
selection are transport-independent. Set `NCCL_CUMEM_ENABLE=0` in both new
physical modes. T0 must validate as `P2P/IPC` direct P2P over NVSwitch;
R3 must validate as SHM with P2P disabled and no NET path. After one paired
smoke, re-run the frozen 384-event E1 communication replay. Only STRONG_GAP
may authorize the existing clean F/K stage. Do not use P2P/CUMEM again,
retune NCCL, rerun C or change rho.

**Bounded cuMem diagnosis complete (2026-10-03): CUMEM_PATH_UNSTABLE.**
All three fresh default-T0 trials selected P2P/CUMEM and timed out at the
90-second bound. The single NCCL_CUMEM_ENABLE=0 diagnostic selected P2P/IPC
and passed four-rank payload validation. This supports path-specific
instability, not an established driver root cause. No E1 retry or model
work was started; the alternate path is not adopted as T0. Target GPUs
0,1,4,5 were released; other GPU jobs were not modified. Read
`CUMEM_PREFLIGHT_RETRY_RESULTS.md` and `cumem_preflight_retry.json`.
Stop here: do not add retries, change the timed baseline or reset drivers.

**Immediate follow-up after `b71199c`: bounded P2P/CUMEM diagnosis and conditional E1 retry.**
Read `experiments/fetch_comm_pareto_p2p_20261002/CUMEM_PREFLIGHT_RETRY.md`.
Run exactly three fresh default-T0 32-KiB preflights plus one diagnostic
preflight with `NCCL_CUMEM_ENABLE=0`, each bounded at 90 s. Commit the
diagnosis first. Only if default T0 passes 3/3 on P2P/CUMEM may the original
E1 trace replay be re-run unchanged in a fresh result root. The cuMem-disabled
path is diagnostic only and must not replace T0. If any default trial is
unstable, stop after diagnosis. Do not start E2/model work unless a subsequent
original-path E1 satisfies the existing STRONG_GAP gate.

**E1 trace-communication stage blocked (2026-10-03): BLOCKED_PREFLIGHT.**
The first T0 32-KiB preflight exceeded its 180-second bound. Communicators
initialized and selected P2P/CUMEM; no validated payload receipt followed.
Last logs show shareable-buffer imports/UDS mapping; root cause is not
established. No OOM: peak process-tree RSS 4.69 GiB, minimum host available
1851.92 GiB. Target GPUs were released. The 384-event input/count validation
and two CPU tests pass, but zero trace timings or model runs were executed.
Read `TRACE_COMM_REPLAY_RESULTS.md` and `trace_comm_preflight_failure.json`.
Do not label this NO_GAP: no ratio exists. E2 remains unauthorized by its
STRONG_GAP gate; stop without an automatic retry, NCCL tuning or server move.

**Immediate follow-up after `b6edeb0`: actual-trace comm gate, then conditional clean F/K.**
Read `experiments/fetch_comm_pareto_p2p_20261002/TRACE_COMM_CLEAN_FK.md`.
First replay the recorded 384 decode-layer dispatch/combine traffic with no
model, expert H2D or cache work. Compare T0/R3 in two counter-ordered passes
and commit this E1 result. Only if R3 is slower in both passes and the median
cumulative ratio is >=1.20x may E2 start. E2 removes heavy per-event validation
from the timed path and runs only F=rho0 and K=rho.25, two repeats per
transport (8 cells). Do not rerun C, add rho values, use Nsight, enable
substitution or tune more NCCL knobs. Commit each stage separately and stop.

**Physical F/K/C pilot complete (2026-10-03): NO_CLEAR_SHIFT.**
All six cells passed trace/cache/action/send-count/token parity (2,592 global
layer events). Descriptive winners differ: T0-K (8.3423 s decode) and R3-F
(6.8918 s), with 5.19%/9.30% margins. However, R3/T0 payload intervals are
0.769x at F and 0.866x at K, and F replay/check CPU time differs by 0.991 s;
the proposed communication-cost mechanism is not supported. No repeat or
follow-up is authorized by this result. Read `PHYSICAL_FKC_RESULTS.md` and
`physical_fkc_validation.json` in the packet; stop for owner review.

**Immediate follow-up after `dc7b099`: six-cell physical F/K/C pilot.**
Read `experiments/fetch_comm_pareto_p2p_20261002/PHYSICAL_FKC_PILOT.md`.
Freeze F/K/C=rho 0/.25/.75 CPU action schedules, then physically replay the
same exact-only one-prefill + eight-decode workload on GPUs 0,1,4,5. Run
exactly: T0-F, R3-C, T0-K, R3-K, T0-C, R3-F. One run per cell. Primary
question is whether the descriptively fastest rho moves between T0 and R3.
Require >=5% margin plus consistent H2D/peer counters for PROMISING_SHIFT.
No automatic repeats, longer trace, Nsight, substitution or final method.
Commit after the six cells or immediately on blocking failure and stop.

**CPU-only replica screen complete (2026-10-03): GO_FOR_OWNER_REVIEW.**
All five budgets are nondominated; F/K/C rho=0/.25/.75. F to C increases
decode H2D by 160.20% and reduces peer activation bytes by 100%. All 2,160
CPU events passed, including 432 independent rho=0 checks. No new GPU/model
run. Read `REPLICA_PARETO_RESULTS.md` and `replica_pareto_validation.json`
in the packet. Stop here; do not rerun or start physical validation without
an owner follow-up. The earlier priorities below are retained as history.

**Immediate follow-up after `ed7f82b`: CPU-only replica Pareto screen.**
NCCL characterization is paused. Read
`experiments/fetch_comm_pareto_p2p_20261002/REPLICA_PARETO_SCREEN.md`.
Reuse the validated exact-only 8-decode trace; no new GPU/model run. Sweep
replica-budget rho={0,0.125,0.25,0.5,0.75} with one common deterministic
first-copy rule, LRU, and greedy current-byte-saving replicas. Primary plane is
decode peer activation bytes vs decode expert H2D bytes. Continue only if at
least three points are nondominated and F/C differ by >=10% on both axes.
Commit the CPU result immediately and stop; do not implement the final method
or physical F/K/C automatically.

**Immediate follow-up after `0c09fae`: exact-only payload capture.**
The existing substituted traces cannot recover exact destination ranks. Run one
R4/B8 P0 capture with substitution off, replication off, LRU, T0, and only
1 prefill + 8 decode forwards. Record actual owner maps and dispatch/combine
send counts; derive payload p50/p90/p99/max. Then run only the 32 KiB, 64 KiB,
256 KiB, 1 MiB and 4 MiB T0/R3 crossover with two counter-ordered passes.
Do not implement replication or Stage 1. If R3 is not >=1.5x slower within the
observed exact-only p90/p99 range, stop synthetic H100 P2P-off work and move to
the real no-NVLink server. Commit the bounded result immediately.

Read `experiments/fetch_comm_pareto_p2p_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

This is the new owner-authorized minimum-scope go/no-go study. Use only GPUs 0,1,4,5; R4/B8/cache30; one prefill + 32 decode forwards; substitution off; LRU fixed. Characterize whether replica budget trades CPU expert H2D against peer activation communication. Compare normal NVSwitch against `NCCL_P2P_DISABLE=1` with SHM fallback left enabled. CPU sweep first; physically validate only F/K/C, two repeats each transport mode (12 primary generations maximum unless the >10% spread rule triggers one targeted repeat). Do not add load balancing, substitution, final weighted objectives, B4/B16 or R8.

**Immediate follow-up after `fad71d2`: payload crossover only.**
Do not start Stage 1/model/replication work. Read
`experiments/fetch_comm_pareto_p2p_20261002/PAYLOAD_CROSSOVER.md`.
Reuse an existing R4/B8 raw-routing trace to compute actual decode rank-pair
payload p50/p90/p99/max, then benchmark only 32 KiB, 64 KiB, 256 KiB, 1 MiB
and 4 MiB per peer under T0 and the validated R3 SHM condition. Two
counter-ordered lightweight passes only. If R3 is not >=1.5x slower within the
real p99 payload range, stop synthetic H100 P2P-off work and move the
communication-sensitive experiment to the real no-NVLink server. Commit the
result immediately; do not tune more NCCL knobs.

**Recovery outcome (2026-10-03): functional, cost-increase gate not met.**
R1 and R2 failed with IB retry errors; R3 passed all-rank payload validation
using `SHM/direct/direct` with P2P_LEVEL=LOC and IB disabled. Stop at that first
success: no loopback retry or further knobs. Its three calibration cells passed,
but peer median was 0.236064 ms versus prior T0 0.336480 ms (0.702x), so there is
no demonstrated communication-cost increase. Stage 1/model/replica work has
not started. Read `TRANSPORT_RECOVERY_RESULTS.md` and
`transport_recovery_result.json` in the packet before further work. The prior
instructions/status below are retained as history; do not automatically rerun
or manufacture a slower condition.

**Immediate transport recovery after `7c881f7`:** do not start the model.
Read `experiments/fetch_comm_pareto_p2p_20261002/TRANSPORT_RECOVERY.md`.
Try only the bounded R1/R2/R3 no-P2P smoke sequence, committing each result.
R1 uses `NCCL_P2P_LEVEL=LOC`; R2 additionally disables GDR; R3 disables IB
and may use Socket/loopback as an explicitly synthetic stress condition. Stop
at the first valid path. If all fail, stop synthetic H100 T1 work and move the
Pareto experiment to the real no-NVLink server rather than exploring more NCCL knobs.

**Stage 0 status (2026-10-03): blocked at T1 transport.** T0 smoke and three
calibration cells passed; T1 selected NET/IB/GDRDMA and failed its first
all-to-all with `IBV_WC_RETRY_EXC_ERR`. See the packet's `RESULTS.md` and
`validation.json`. No model/replica runs have started. Resolve the transport
gate before primary timing; never label this observed path verified SHM.
The owner requests immediate incremental commits at stage boundaries and on
failures, rather than waiting for the whole packet to finish.

# mgo_v2 coding instructions

## New priority — exact rank-demand oracle and GPU critical-path validation (2026-10-02)

Read `experiments/rank_demand_oracle_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

**Owner scope reduction (2026-10-02):** the latest user instruction requests the
minimum sufficient experiment set. Read `scope_amendment.json` in that packet.
Preserve the completed six-repeat B4 results; B8/B16 use two repeats per policy
(30 primary generations total). Controller follow-up is B8 only, one fresh
C0/C1/C2 generation per policy (9 total), full decision/cache/token parity and
separate diagnostics. Do not expand controller timing to B4/B16. This amendment
overrides the larger repetition/expansion matrix in the earlier plan.

Use only physical GPUs **0,1,4,5** as R4. Compare Balanced Random, existing Hungarian-current, and an exact diagnostic **rank-demand oracle** that ignores communication and minimizes the busiest rank's current effective expert-token rows under the same hard admission-count quota. Oracle assignments are generated in an untimed planning pass and replayed frozen for primary E2E/TPOT, with exact plan/cache/token parity required; solver time is reported separately. Run B4/B8/B16, 64 decode forwards, six uninstrumented repetitions using all policy-order permutations once per batch. After timing, profile only R4/B8 to measure actual max-rank expert GPU time, NCCL, H2D and layer completion. Do not add a joint communication+load policy before this headroom study is complete.

**Oracle packet status: complete.** The owner-reduced 30 generations and three full B8 profiles passed at `experiments/rank_demand_oracle_20261002/{RESULTS.md,validation.json}`. Do not rerun the packet. Continue only the separately authorized reduced controller-overhead follow-up.

**Controller follow-up status: complete.** The reduced B8 packet passed at
`experiments/controller_overhead_20261002/{RESULTS.md,validation.json}`: nine
primary generations, 9,360 captured CPU differential events and two short P1
profiles. C1/C2 remain opt-in. Do not automatically rerun, add repetitions,
expand to B4/B16 or change placement policies. Single-sample timing is descriptive.

## New priority — admission trajectory/controller breakdown (2026-10-02)

The physical locality study is complete at `e61758e`. Before changing the method, read `experiments/admission_trajectory_controller_breakdown_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

**Status: the bounded trajectory packet is complete.** Read `experiments/admission_trajectory_controller_breakdown_20261002/{RESULTS.md,validation.json,implementation_audit.md}`. Six R4 diagnostics and 60 CPU replays passed; do not automatically rerun the packet. Any method change or controller optimization remains a separate owner-reviewed follow-up.

Only four GPUs are available for new work. Add diagnostic-only subcomponent timers/counters to `plan_layer`, capture cache/fetch/eviction trajectories, and run matched-raw-demand controller replay. GPU scope is exactly six **R4** diagnostic runs: Random/Current at local B4, B8 and B16, always on physical GPUs 0,1,4,5. **Do not launch any R8 job.** Existing R8 results at `e61758e` are retrospective context only. Existing five-repeat E2E results remain the performance evidence. Do not tune or redesign admission, optimize controller code, add token-load constraints, or revive same+path before this mechanistic packet is complete.

## New priority — local/remote TPOT and E2E impact study (2026-10-01)

Read `experiments/local_remote_e2e_impact_20261001/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}` before new timing work.

Use the validated mgo_v2 physical offloading runtime. Stage A isolates peer-communication sensitivity with resident experts and zero H2D inside timed ranges. Stage B measures uninstrumented five-repeat TPOT/E2E for Random vs Hungarian-current vs Hungarian-same+path at R4/R8 cache30 with 64 fixed decode steps. Freeze same-layer support64/alpha=.25 and path support64/eta=.5 explicitly; do not use the historical alpha=1 default. Profile only after uninstrumented timing and do not retune from timing outcomes.

The mgo_v2 directory is the current implementation target.

## Authority

- mgo_v2 GlobalExpertController is the only logical cache authority.
- Do not re-enable legacy DeviceMapManager, ExpertPrefetcher cache replacement,
  ExpertCache eviction, or Archer autonomous sparse eviction.
- Preserve no-replication/no-migration semantics unless an experiment
  explicitly changes them.

## Frozen research defaults

- expert-level substitution;
- gate protection threshold .20;
- similarity threshold .65;
- Gate history W=128;
- diversity eviction k=1, lambda=2;
- hard-balanced admission quota;
- no CPU attention.

## Correctness before performance

Order of work:

1. CPU policy tests;
2. exact R1/R4 EP parity;
3. cache state parity;
4. substitution parity;
5. eviction parity;
6. admission parity;
7. R8;
8. low-level overlap and D2D optimization;
9. final timing.

Never report performance from a run that still uses the old RPC expert path or
has two residency controllers active.
