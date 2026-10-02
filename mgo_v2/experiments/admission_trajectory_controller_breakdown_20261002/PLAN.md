# PLAN — Why does admission win or lose?

Date: 2026-10-02  
Status: prospective analysis plan. Do not launch GPU jobs from this commit.

## 1. Starting evidence

Use the completed physical study at `e61758e` as the fixed starting point.

The completed study provides retrospective context:

| Cell | Hungarian-current vs Random E2E | Role |
|---|---:|---|
| R8/B4 | ~38% faster | historical strong positive case |
| R8/B8 | ~18% slower | historical negative case |
| R4/B8 | ~5% slower | existing four-GPU negative case |
| R8/B16 | ~8% faster, noisy | historical secondary case |

Only four GPUs are available now. Therefore **all new physical diagnostics use R4 on GPUs 0,1,4,5**. Existing R8 measurements may be reanalyzed descriptively but must not trigger any R8 launch.

The new four-GPU batch sweep uses R4/B4, R4/B8 and R4/B16. This gives global batches 16, 32 and 64 and tests whether the admission/controller trajectory changes systematically with per-rank demand.

The completed study already showed that remote-pair reduction alone is insufficient: Hungarian-current reduces remote pairs while E2E direction can differ.

## 2. Questions

### Q1 — Where does controller time go?

`DistributedMoERuntime.controller_seconds` currently times the entire:

[
GlobalExpertController.plan_layer()
]

after global route metadata has already been gathered.

Break it into:

[
T_{plan}
=
T_{history}
+
T_{substitution}
+
T_{effective route}
+
T_{admission}
+
T_{eviction}
+
T_{cache mutation}
+
T_{bookkeeping}.
]

For Hungarian admission, further split:

[
T_{admission}
=
T_{token/base prep}
+
T_{cost matrix}
+
T_{assignment}.
]

For Coverage eviction, further split:

[
T_{eviction}
=
T_{candidate}
+
T_{coverage sync}
+
T_{gate/damage ranking}
+
T_{victim selection}.
]

The purpose is to determine whether the large Random-vs-Current controller differences are caused by the admission solver itself or by the downstream state trajectory.

### Q2 — Does admission alter the future cache trajectory enough to explain the win?

Measure, per global layer event and cumulatively:

- residual exact misses;
- admissions;
- evictions;
- reloads;
- exact hits and substitution hits;
- eviction candidate count;
- changed coverage layers;
- cache-owner changes;
- per-rank free-slot pressure;
- physical H2D fetch count/bytes;
- remote/local token-rank pairs;
- rank token CV and max/mean load.

Track every admitted expert to its **next raw demand**:

- next-demand event distance;
- whether it survives resident until next demand;
- whether next demand is exact-hit, substitution-covered, or reload;
- whether it changes owner before next demand.

Primary trajectory metrics:

[
NextUseSurvival =
rac{# admitted experts still resident at next demand}
{# admitted experts with a later demand}
]

and

[
ReloadPerAdmission =
rac{# reloads}{# admissions}.
]

Also report eviction/admission events per decode step and physical H2D per decode step.

### Q3 — Is rank-load skew offsetting communication savings?

The existing hard quota balances the **number of incoming experts**, not routed-token/GEMM work.

For every event retain:

- expert-count quota per rank;
- routed token count per rank;
- effective expert-route count per rank;
- max/mean token load;
- token-load CV;
- per-rank expert GEMM token rows if available without changing execution.

Compare communication savings against load skew:

[
Delta RemotePairs
quad	ext{vs}quad
Delta MaxRankTokenLoad.
]

Do not add a load-aware policy yet. This stage only establishes whether the missing constraint is real.

### Q4 — How much of the communication saving is directly visible in GPU time?

The completed Stage-A study found only a small/non-monotonic resident-only latency response on NVSwitch. Reuse that evidence.

If a new profile is needed, compare Random vs Hungarian-current, not same+path, using only R4.

Preselect:

- R4/B4;
- R4/B8.

Use physical GPUs 0,1,4,5. Measure NCCL kernel union, H2D, GPU idle/wait and expert compute. Do not infer additive percentages from overlapping intervals.

## 3. Instrumentation design

### 3.1 Zero-overhead default

All new detailed instrumentation must be **disabled by default**.

The normal runtime path must preserve the validated output/policy behavior and existing timing results.

Use an explicit diagnostic flag such as:

`MGO_CONTROLLER_BREAKDOWN=1`

or an equivalent benchmark CLI flag.

### 3.2 Controller timing

Use `time.perf_counter_ns()` around diagnostic subregions.

Do not call CUDA synchronization inside controller sub-timers.

Record:

- per-event raw timings;
- cumulative timings;
- number of calls for each subregion.

For eviction, aggregate time across all victim selections in an event and count how many victims were selected.

### 3.3 Operation counters

Add counters rather than reconstructing expensive quantities after the fact.

Examples:

- number of `DiversityEviction.choose()` calls;
- candidate-set sizes;
- number of `_sync_coverage()` calls;
- number of resident additions/removals observed by coverage sync;
- number of layers whose coverage arrays were recomputed;
- Hungarian matrix shape `incoming × world`;
- cost-build and `linear_sum_assignment` time separately.

### 3.4 No scientific semantic change

Instrumentation must not change:

- routing;
- substitution;
- admission decisions;
- victim decisions;
- cache capacity;
- random seeds;
- generated tokens;
- hard quotas.

Add parity tests proving diagnostic on/off produces byte-identical policy decisions and generated tokens on the smoke workload.

## 4. Three analysis modes

Do not rely on only one kind of run.

### Mode A — retrospective existing-result analysis

Before any GPU work, reanalyze the existing `e61758e` receipts.

For Random and Hungarian-current in the already-completed R8/B4, R8/B8, R8/B16, R8/B32 and R4/B8 results:

- remote pairs;
- physical H2D;
- fetches/reloads;
- controller time;
- rank token CV;
- E2E/TPOT.

Produce a compact table and correlations only as descriptive evidence.

No claim of causality from this mode.

### Mode B — diagnostic physical rerun

**World size is fixed to R4. No R8 launch is permitted.**

Run only:

1. R4/B4 — global batch 16;
2. R4/B8 — global batch 32, existing negative reference;
3. R4/B16 — global batch 64.

Policies:

- Balanced Random;
- Hungarian Current.

Use physical GPUs **0,1,4,5** for every run and the same checkpoint/workload/cache30/substitution/Coverage settings as `e61758e`.

Protocol:

- 64 decode steps;
- one detailed diagnostic run per cell/policy after correctness gates;
- no profiler during controller-breakdown collection;
- save every per-event controller/trajectory record;
- do not use diagnostic wall time as a new speedup result.

The existing five-repeat uninstrumented R4/B8 result remains the prior performance reference. R4/B4 and R4/B16 are diagnostic batch-scaling probes, not new publication speedup claims.

### Mode C — frozen-route controller replay

Capture raw global router metadata from the diagnostic runs:

- layer;
- phase/decode step;
- origin ranks;
- selected experts;
- routing weights;
- full router probabilities.

Raw routes are stored on the server, not committed if large.

Then run single-process CPU controller replay in two forms.

#### Own-trajectory replay

Replay each policy on its own captured raw-route stream.

Purpose: reproduce and decompose the controller work without distributed OS/GPU timing noise.

#### Matched-demand replay

Take the Random raw-route stream for a cell and feed the **same raw demand** to:

- Balanced Random controller;
- Hungarian Current controller.

Each controller keeps its own cache/substitution/eviction state, but raw model demand is held fixed.

Repeat the reverse direction using the Current raw-route stream as a sensitivity check.

This is a systems counterfactual, not a model-quality run.

It answers whether the future cache/controller divergence is caused by the admission policy under matched raw demand rather than by different generated hidden-state trajectories.

Run each replay at least 5 times and report median subcomponent CPU times.

## 5. Event-aligned trajectory analysis

For each matched replay, align events by:

[
(step, layer, phase).
]

Compute cumulative deltas:

[
Delta Fetch(k)=
sum_{ile k}(Fetch_{Current,i}-Fetch_{Random,i})
]

[
Delta Evict(k)=
sum_{ile k}(Evict_{Current,i}-Evict_{Random,i})
]

[
Delta Controller(k)=
sum_{ile k}(T_{Current,i}-T_{Random,i}).
]

Also plot:

- cumulative remote-pair delta;
- cumulative reload delta;
- cumulative coverage-sync time delta;
- cumulative eviction time delta.

The key question is **when the trajectories diverge**.

Compare when R4/B4, R4/B8 and R4/B16 begin to diverge. If one batch regime saves future eviction/reload/controller work while another accumulates extra work, that is direct evidence that admission value depends on the evolving cache/controller trajectory rather than communication alone.

## 6. Counterfactual cache-state checks

At selected divergence events, snapshot the logical cache state.

For each policy record:

- resident expert keys by rank;
- exact execution owners;
- next-demand distance of residents;
- gate score;
- coverage damage;
- whether each resident becomes exact-hit/substitute anchor before eviction.

Do not create a new victim-aware admission method here.

The purpose is to identify concrete examples such as:

> Current placed expert e on rank r; that resident survived N events and served M future exact/substitute demands, while Random evicted/reloaded the same source.

At least five positive and five negative event examples should be retained when available.

## 7. Controller implementation audit

Separately from scientific trajectory effects, quantify avoidable implementation overhead.

Inspect and measure:

1. every rank independently recomputes the same deterministic global plan;
2. repeated `cache.keys_on_rank()` list construction;
3. Coverage candidate/rank calculations;
4. Python loops in substitution/admission;
5. repeated conversion/allocation of NumPy arrays and sets;
6. Hungarian cost construction versus SciPy solve time.

Do **not** optimize first.

First publish:

[
	ext{algorithmic work counts}
]

and

[
	ext{implementation time per component}.
]

Only after owner review may a separate optimization stage vectorize/C++-move/cache/precompute these operations.

## 8. Hypotheses and decision rules

### H1 — future trajectory matters

Supported if, under matched raw demand, Hungarian-current changes future miss/reload/eviction behavior enough that cumulative downstream controller/H2D work differs materially from Random.

### H2 — admission solver itself is not the main controller cost

Supported if cost-build + assignment is a minority of total `plan_layer` time while eviction/coverage/substitution/cache work explains most of the Random/Current gap.

### H3 — expert-count hard balance is insufficient

Supported if the communication-aware policy consistently increases max-rank token work/CV in negative cells, and the negative-cell slowdown is temporally aligned with that skew after controlling for controller time as far as the trace allows.

Do not claim H3 solely from aggregate CV.

### H4 — direct NVSwitch byte saving is secondary

Supported if remote-pair/payload reduction is substantial but matched resident-only/profiler GPU communication time changes much less than E2E/controller changes.

This is already suggested by `e61758e`; new profiles are optional and diagnostic.

## 9. Bounded GPU matrix

Primary GPU diagnostic matrix:

| World | Local B | Global B | Policy |
|---:|---:|---:|---|
| 4 | 4 | 16 | Random |
| 4 | 4 | 16 | Hungarian-current |
| 4 | 8 | 32 | Random |
| 4 | 8 | 32 | Hungarian-current |
| 4 | 16 | 64 | Random |
| 4 | 16 | 64 | Hungarian-current |

Every physical run must use GPUs **0,1,4,5**. **Do not launch any R8 job.**

No same+path tuning. No new admission weights. No cache-ratio sweep.

Optional profiles are gated on the controller analysis and preselected to Random/Current for R4/B4 and R4/B8 only.

## 10. Required artifacts

Create under this experiment folder:

- `IMPLEMENTATION.md`
- `measurement_manifest.json`
- `controller_breakdown.csv/json`
- `trajectory_events.csv` or a compact server-side trace plus summary CSV;
- `trajectory_summary.csv/json`
- `next_use_survival.csv/json`
- `matched_replay_summary.csv/json`
- `state_examples.md`
- `implementation_audit.md`
- `RESULTS.md`
- validation/provenance/hash receipts.

Do not commit giant raw route traces or Nsight traces.

## 11. Stop condition

Stop after:

1. instrumentation parity tests;
2. retrospective analysis;
3. six four-GPU diagnostic runs;
4. own-trajectory and matched-demand CPU replays;
5. controller/trajectory result packet.

Do not automatically:

- change the admission objective;
- add token-load constraints;
- optimize controller code;
- retune Coverage;
- revive same+path;
- add replication or migration.

Those are follow-up design choices that require owner review after the causal/mechanistic analysis.
