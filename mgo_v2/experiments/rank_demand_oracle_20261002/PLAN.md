# PLAN — Demand-balanced oracle placement and TPOT/E2E remeasurement

Date: 2026-10-02
Status: prospective experiment plan. Do not launch GPU work from this commit.

## 1. Starting point

The completed trajectory study at `86fc58a` showed, under matched raw demand:

- Hungarian-current reduces remote token-rank pairs by about 12–17%;
- matched-demand fetch/reload changes are nearly zero;
- assignment cost-build + Hungarian solve is <1% of controller time;
- communication-aware placement increases the decode sum of busiest-rank planned expert GEMM rows by about 14–29%.

The missing causal check is whether that expert-row skew turns into actual GPU critical-path time and whether a demand-balanced rank placement reduces TPOT/E2E.

## 2. Hardware and fixed system

Use only physical GPUs:

[
0,1,4,5
]

as logical ranks 0..3.

No R8 launch.

Freeze:

- Qwen3-30B-A3B-Instruct-2507, BF16;
- cache 30%;
- expert-level substitution gate=.20 / similarity=.65;
- Coverage eviction W128/k1/lambda2;
- no replication or migration;
- native expert-order arithmetic;
- 64 decode forwards after one prefill;
- hard admission-count quotas identical to the existing controller.

The oracle changes **only the rank assignment of residual exact misses**.

## 3. Policies

Compare exactly three policies.

### P0 — Balanced Random

Existing deterministic `BalancedRandomAdmission(seed=42)`.

### P1 — Hungarian Current

Existing communication-aware assignment.

This remains the communication baseline.

### O0 — Exact Rank-Demand Oracle

O0 ignores communication completely.

At a global layer event, after substitution is fixed, define:

- (d_e): number of effective expert-token rows requiring execution of incoming exact-miss expert (e);
- (B_r): current expert-token rows already assigned to rank (r) by execution experts resident before this event;
- (x_{e,r}in{0,1}): assign incoming expert (e) to rank (r);
- (q_r): the same hard balanced admission-count quota used by P0/P1.

Predicted rank work is:

[
L_r=B_r+sum_e d_e x_{e,r}.
]

Solve:

[
min z
]

subject to:

[
sum_r x_{e,r}=1,
]

[
sum_e x_{e,r}=q_r,
]

[
L_rle z.
]

Existing resident expert owners are fixed. No migration or replication is allowed.

This is an **exact current-demand min-max oracle within the existing admission action space**. It does not minimize remote pairs.

Use an exact MILP/HiGHS path available through the installed SciPy stack. Add a deterministic second solve/tie-break constrained to the optimal (z) so repeated planning produces identical assignments.

If exact solve fails or times out for any event, do not silently fall back to a heuristic. Mark the oracle planning pass invalid.

## 4. Why the oracle is replayed rather than timed inline

Exact MILP solve time is not part of the data-plane question.

Use two phases.

### 4.1 Oracle planning pass

Run the full model once per workload cell with O0 active.

For every event record:

- raw global routes;
- substitution decision;
- incoming exact misses;
- (d_e);
- preowned base load (B_r);
- exact assignment;
- achieved (L_r) and (z);
- eviction/cache state hash;
- generated tokens;
- solver time/status.

This pass creates a deterministic event-by-event oracle assignment trace.

### 4.2 Frozen-oracle timed replay

Rerun the same model/workload from empty cache/history.

At each event, inject the precomputed oracle admission assignment instead of solving again. All other controller operations remain online and unchanged.

Require exact equality to the planning pass for:

- raw route hash;
- substitution mapping;
- incoming expert set;
- assignment legalities/quotas;
- victims/cache-state hash;
- generated tokens.

If any event diverges, the timed oracle replay is invalid.

This gives an **upper-bound E2E/TPOT headroom** for demand-aware placement without charging the expensive exact oracle solve.

Report oracle solve time separately; never present O0 as a deployable online speedup.

## 5. Stage A — actual GPU critical-path remeasurement

Purpose: test whether the prior expert-row imbalance proxy corresponds to GPU time.

Primary cell:

[
R4/B8,quad global batch=32.
]

Profile P0, P1 and O0 after all uninstrumented timing is complete.

Add diagnostic-only NVTX/range instrumentation around:

1. route metadata collective;
2. H2D expert fetch;
3. dispatch;
4. rank-local expert execution;
5. combine;
6. complete MoE layer.

Do not use profiled wall time as TPOT/E2E.

From Nsight derive, per decode step and layer:

- each rank's expert-execution GPU interval;
- (max_r T_{expert,r});
- each rank's NCCL dispatch/combine interval;
- max-rank NCCL interval;
- H2D copy interval;
- MoE-layer completion interval.

Also record per rank:

- effective expert-token rows;
- distinct executing experts;
- GEMM kernel count.

Primary causal-style check:

[
Delta(max_r ExpertRows)
quad	ext{vs}quad
Delta(max_r ExpertGPUTime).
]

If P1 increases expert rows but not expert GPU time, the row-count proxy is insufficient.

## 6. Stage B — uninstrumented TPOT/E2E comparison

Use three R4 workload cells:

| Local B | Global B |
|---:|---:|
| 4 | 16 |
| 8 | 32 |
| 16 | 64 |

Policies:

- P0 Balanced Random;
- P1 Hungarian Current;
- O0 frozen Rank-Demand Oracle.

Each condition uses:

- 64 decode forwards;
- empty expert cache/history at repeat start;
- identical workload/checkpoint;
- no profiler/CUDA-event instrumentation;
- all GPU jobs sequential.

### 6.1 Six balanced repetitions

Use **6 repeats** per cell/policy.

For each batch, rotate through all six permutations of the three policy orders exactly once. This removes the fixed Random-then-Current order used by earlier measurements.

Publish every repeat and the median/min/max.

Primary metrics:

- TPOT;
- full generation time;
- output tokens/s.

Required secondary metrics:

- TTFT;
- physical H2D bytes/fetches/reloads;
- remote token-rank pairs and fraction;
- controller wall time;
- rank token CV;
- sum/max of planned expert-token rows.

## 7. Oracle headroom metrics

For each event compute:

[
LoadReduction=
1-rac{max_r L_r^{O0}}{max_r L_r^{P0}}.
]

Also compare O0 to P1.

At cell level report:

[
TPOTReduction_{O0/P0}=1-rac{TPOT_{O0}}{TPOT_{P0}}
]

and similarly for E2E.

The key test is whether reducing max-rank expert demand produces lower TPOT/E2E even if remote communication becomes worse.

## 8. Interpretation matrix

Four outcomes are possible.

### Case A

O0 lowers max-rank expert GPU time and TPOT/E2E.

Interpretation: rank expert-demand balance is a real critical-path lever.

### Case B

O0 lowers predicted expert rows but not expert GPU time.

Interpretation: expert-row count is not a sufficient compute proxy; inspect expert grouping/kernel efficiency.

### Case C

O0 lowers expert GPU time but E2E does not improve.

Interpretation: another component (Coverage/controller, H2D, NCCL, synchronization) dominates.

### Case D

O0 does not lower predicted max-rank work in practice.

Interpretation: existing hard quota/preowned state leaves little balancing headroom.

Do not tune the oracle objective after seeing results.

## 9. Communication is deliberately not optimized

O0 has **no communication term**.

Always report its remote pairs and peer payload beside TPOT.

This is important:

- if O0 is faster while communication is worse, compute balance is stronger evidence;
- if O0 is slower despite balanced compute, communication/synchronization remains important;
- if P1 and O0 each help in different regimes, the eventual method likely needs a joint objective or a hard load constraint.

Do not implement that joint method in this experiment.

## 10. Cache/fetch controls

Because rank assignment changes victims and future cache state, retain:

- fetches;
- reloads;
- next-use survival;
- H2D bytes.

However, this experiment does not claim future-residency optimization.

If O0 changes H2D materially, state explicitly that E2E differences are not pure compute-balance effects.

## 11. Correctness gates

Before GPU timing:

1. CPU tests for exact oracle feasibility and deterministic tie-break;
2. exhaustive brute-force parity on small synthetic assignments;
3. hard-quota/capacity/pinned-owner invariants;
4. frozen-oracle replay reproduces planning-pass assignment/cache/token hashes;
5. diagnostic instrumentation on/off preserves outputs.

No event may fall back from exact oracle to heuristic.

## 12. Controller-overhead repair and clean remeasurement

This is a **separate follow-up stage after the placement/oracle measurements above**. If Stage A/B has already started from the previous plan commit, let it finish unchanged. Do not mix controller changes into the primary P0/P1/O0 placement comparison.

The completed controller breakdown established that the current runtime has a large host-side planning bottleneck:

- Coverage ranking accounts for roughly **69.5–71.0%** of instrumented controller time;
- each victim selection visits about **448–452 candidates**;
- the measured R4 traces perform roughly **20.6M–42.1M candidate visits** across B4–B16;
- repeated `keys_on_rank()`, resident list/set construction, candidate score/rank construction, `_sync_coverage()`, and full cache-consistency scans contribute additional Python work;
- cost construction plus Hungarian assignment is at most **0.96%** of controller time, so optimizing the assignment solver is not the priority;
- every rank currently receives the same global routing metadata and independently computes the same global plan, duplicating logical planning work across ranks.

The goal of this stage is to remove implementation overhead **without changing any controller decision** and then remeasure whether the placement/load effect becomes more visible in TPOT/E2E.

### 12.1 C0 — frozen baseline controller

Use the exact existing implementation as the baseline. Preserve the already-collected P0/P1/O0 placement/oracle results as the primary placement evidence. For a fair C0/C1/C2 controller comparison, fresh matched C0 control repeats may be run in the controller-overhead stage, but they must be reported separately and must not overwrite the primary placement/oracle packet.

Record, per event and per run:

- total controller wall time;
- Coverage victim ranking time;
- admission construction/assignment time;
- substitution time;
- cache/history synchronization time;
- candidate visits;
- cache-key/list materializations;
- plan/cache hashes.

### 12.2 C1 — decision-equivalent local controller optimization

Optimize the current controller implementation while keeping each rank's planning semantics unchanged.

Priority order:

1. **Coverage victim selection**
   - avoid materializing the full rank cache for every victim;
   - reuse resident/pinned views within an event;
   - cache or incrementally maintain gate/coverage quantities that are currently rebuilt for every candidate;
   - avoid repeated Python sorting/scoring when the relevant state has not changed.

2. **Resident/list/set construction**
   - remove repeated `keys_on_rank()` scans and redundant global resident-set reconstruction;
   - update affected resident/layer state incrementally where possible.

3. **Validation-only scans**
   - move expensive full-cache consistency scans out of the timed production path or gate them behind a debug/validation option;
   - retain equivalent correctness tests outside primary timing.

C1 must preserve, event by event:

- substitution mapping;
- admission assignment;
- selected victim;
- cache state;
- effective routes;
- generated tokens.

Any semantic difference invalidates C1 as an overhead-only optimization.

### 12.3 C2 — remove replicated global planning

After C1 parity passes, measure a second runtime variant in which one designated planner rank computes the deterministic global plan once and broadcasts a compact plan/decision payload to the other ranks.

Requirements:

- all ranks still contribute the same required global routing metadata;
- planner output must be sufficient for every rank to execute exactly the C1 plan;
- the broadcast payload and synchronization cost must be measured explicitly;
- event-level plan/cache/token hashes must match C1 exactly;
- no policy coefficient, cache rule, substitution decision, quota, placement objective, replication or migration change is allowed.

C2 tests whether replacing **R copies of identical CPU planning** with one planning pass plus a small collective improves the real critical path.

### 12.4 Controller optimization timing matrix

After parity/correctness gates, remeasure the primary R4/B8 cell first.

For each of P0, P1 and O0 frozen placement traces compare:

- C0 baseline controller;
- C1 local optimized controller;
- C2 single-planner controller.

Use uninstrumented fixed-work generation for TPOT/E2E. Use separate diagnostic runs for subcomponent controller timings.

If R4/B8 shows a material controller reduction with stable outputs, expand the uninstrumented comparison to R4/B4 and R4/B16.

Use the same workload/checkpoint/cache/substitution/Coverage settings and the same balanced policy-order discipline as Stage B. Do not retune placement coefficients after seeing the controller results.

### 12.5 Primary controller-overhead metrics

Report:

[
ControllerReduction = 1 - \frac{T_{controller}^{optimized}}{T_{controller}^{C0}}
]

and separately:

- TPOT;
- generation time;
- controller time / generation time;
- Coverage ranking time;
- candidate visits per victim;
- planner broadcast time and bytes for C2;
- GPU expert/NCCL/H2D metrics to verify that a controller-only change did not alter device work.

The important interpretation is:

- if controller time falls and TPOT/E2E falls with identical GPU work, the previous host planning path was masking placement gains;
- if controller time falls but TPOT/E2E barely changes, host planning was largely overlapped or outside the critical path;
- if C1 helps but C2 does not, replicated planning was not worth the added synchronization;
- if C2 helps, the production controller should not compute the identical global plan independently on every rank.

### 12.6 Correctness gates for controller optimization

Before any optimized timing claim:

1. run existing CPU tests;
2. add event-level C0/C1/C2 differential tests;
3. require identical assignments, victims, cache hashes, substitution mappings and generated tokens;
4. verify debug-only consistency checks still pass offline;
5. verify C2 payload decode reproduces the planner rank's exact plan on every rank;
6. keep diagnostic instrumentation outside the uninstrumented timing path.

The controller optimization is a **runtime repair**, not a new placement policy. Report its gain separately from the P0/P1/O0 placement comparison.

## 13. Required artifacts

Create:

- `IMPLEMENTATION.md`
- `measurement_manifest.json`
- `oracle_solver_validation.json`
- `oracle_planning_summary.csv/json`
- `e2e_repeats.csv/json`
- `e2e_summary.csv/json`
- `rank_load_summary.csv/json`
- `gpu_phase_profile.csv/json`
- `RESULTS.md`
- validation/provenance/hash receipts
- figure-support CSVs.

Recommended figures:

1. max-rank expert rows vs measured expert GPU time;
2. TPOT/E2E for P0/P1/O0 across B4/B8/B16;
3. remote-pair change versus max-rank expert-load change;
4. per-step max-rank expert load for R4/B8.

## 14. Stop condition

Stop the placement/oracle stage after:

1. oracle correctness tests;
2. planning traces;
3. 54 uninstrumented generations (3 batches × 3 policies × 6 repeats);
4. three R4/B8 posthoc profiles;
5. the placement/oracle result packet.

Then execute the controller-overhead stage in Section 12:

6. C0/C1/C2 decision-parity tests;
7. R4/B8 controller timing + uninstrumented comparison for P0/P1/O0;
8. expand to B4/B16 only if the R4/B8 controller optimization materially reduces controller time without semantic drift;
9. publish a separate controller-overhead result packet.

Do not automatically:

- add communication to O0;
- add a soft/hard load-aware production policy;
- retune Coverage policy semantics;
- change substitution;
- add replication/migration.

Those remain follow-up method decisions after owner review.
