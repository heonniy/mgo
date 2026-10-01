# PLAN — Local/remote communication sensitivity and E2E impact

Date: 2026-10-01
Status: prospective plan only. This commit does not launch GPU jobs.

## 1. Questions

### Q1 — communication sensitivity

With expert compute and expert identities fixed, how much does MoE latency change as the fraction of expert service that is remote rather than local increases?

### Q2 — full-system C benefit

With real CPU expert offloading enabled, how much does communication-aware admission reduce:

- TPOT,
- complete generation wall time,
- TTFT,
- peer communication,

relative to the same substitution/eviction stack using balanced-random admission?

Q1 isolates peer communication. Q2 measures the end-to-end result, which may also change future cache residency, H2D fetches, controller work and synchronization.

## 2. Frozen runtime and model

Use this repository/branch and bind every run to the exact runtime fingerprint.

Validated baseline commit:

`40a63d8ba22d16684a358d4546bdae4783931e5e`

Model:

- Qwen3-30B-A3B-Instruct-2507
- BF16
- 48 MoE layers
- 128 routed experts/layer
- top-8
- hidden size 2048
- expert intermediate size 768
- physical expert size approximately 9 MiB.

Freeze system semantics:

- corrected expert-level substitution;
- gate protection threshold = .20;
- similarity threshold = .65;
- Coverage eviction W=128 / k=1 / lambda=2;
- no replication;
- no migration;
- hard-balanced admission quota;
- persistent cache within a measured run;
- native BF16 arithmetic and expert-order accumulation.

Primary C policy uses the sub-moe-selected affinity settings, not the historical mgo default:

- same-layer support = 64;
- same-layer alpha = .25;
- path support = 64;
- path eta = .5.

The launcher must record these explicitly. Never silently fall back to alpha=1.

## 3. Communication definitions

For token t with origin rank o(t), let D(t) be the set of distinct ranks that execute at least one selected expert for t.

Local token-rank pair:

[
(t,o(t)) quad 	ext{if }o(t)in D(t)
]

Remote token-rank pair:

[
(t,r),quad rin D(t), r
e o(t)
]

Primary locality metric:

[
RemotePairFraction=
rac{N_{remote token-rank pairs}}
{N_{local token-rank pairs}+N_{remote token-rank pairs}}
]

and:

[
LocalPairFraction=1-RemotePairFraction.
]

Also report route-level local fraction separately. Do not call route-local fraction the communication fraction because dispatch is deduplicated per token/destination-rank.

## 4. Stage A — resident-only local/remote sensitivity

Purpose: isolate peer communication cost from CPU->GPU expert fetch.

### 4.1 Workload

Use representative real Qwen3 MoE layer events from:

- R4 / local batch 8,
- R8 / local batch 8.

Select deterministic low/median/high active-expert events.

Before the timed range, ensure every expert needed by the event is resident. Assert zero expert H2D inside every timed range.

### 4.2 Controlled owner maps

For the exact same tokens, routed experts and expert arithmetic, construct hard-balanced owner maps spanning a broad locality range:

1. `LOCAL_MAX`: minimize exact deduplicated remote token-rank pairs;
2. `LOCAL_HIGH`: high-locality intermediate map;
3. `LOCAL_MID`: intermediate map;
4. `LOCAL_LOW`: low-locality map;
5. `REMOTE_MAX`: maximize remote token-rank pairs subject to the same hard quota.

Use the same solver family/assignment complexity across the five maps.

Report the actual achieved RemotePairFraction for every map. The labels are not claims of exact 75/50/25 percentages.

### 4.3 Timing

For each event/map:

- warm up;
- >=20 timed iterations;
- measure actual MoE layer wall time;
- measure NCCL dispatch+combine CUDA interval;
- submitted peer payload bytes;
- local and remote token-rank pairs;
- rank token CV and max/mean rank load.

Primary relationship:

[
RemotePairFraction ightarrow MoELayerLatency
]

Secondary relationship:

[
RemotePairFraction ightarrow NCCLCriticalInterval.
]

This stage is the direct communication-sensitivity evidence.

## 5. Stage B — real offloading TPOT/E2E

This stage enables physical CPU expert offloading.

### 5.1 Primary cells

R4 anchor:

- ranks = 4;
- local batch = 8;
- global batch = 32;
- cache = 30%.

R8 anchor:

- ranks = 8;
- local batch = 8;
- global batch = 64;
- cache = 30%.

Matched-global-batch control:

- ranks = 8;
- local batch = 4;
- global batch = 32;
- cache = 30%.

### 5.2 Admission policies

Keep substitution and Coverage eviction identical in every row.

Compare:

1. `C0_random`: balanced-random admission;
2. `C1_hungarian_current`: hard global assignment using current communication;
3. `C2_hungarian_same_path`: Hungarian + same-layer co-activation + adjacent-layer path.

H3 swap is allowed only as one diagnostic condition. It is not the primary method because the completed server validation found large controller/critical-path cost despite fewer remote pairs.

### 5.3 Generation length and repeats

Primary:

- 64 fixed decode steps;
- 2 untimed warmup steps;
- 5 measured repeats per condition.

Confirmation if resources remain:

- 128 fixed decode steps for C0 and C2 only.

Before each measured repeat:

- reset logical/physical expert residency and policy history;
- keep dense model weights and CUDA allocator/kernel caches loaded;
- no concurrent GPU workload.

Conditions run sequentially.

### 5.4 Timing rule

**Uninstrumented runs define TPOT and E2E.**

For every repeat record:

- TTFT;
- TPOT;
- complete generation wall time;
- output tokens/s;
- controller wall time;
- physical expert H2D bytes;
- fetches and reloads;
- LocalPairFraction / RemotePairFraction;
- remote token-rank pairs;
- submitted peer payload bytes;
- rank token CV.

Do not average away repeats. Publish all five.

### 5.5 Posthoc profile

After uninstrumented timing is complete, profile exactly one representative C0 and one C2 repeat with the already validated Nsight configuration.

Report:

- actual 9 MiB expert H2D copies;
- H2D/GEMM overlap;
- NCCL kernel critical intervals;
- controller CPU time;
- GPU wait/idle intervals where available.

Profiled wall time is diagnostic only and must not replace the uninstrumented TPOT/E2E number.

## 6. Main comparisons

For each R4/R8 anchor:

[
TPOTSpeedup=rac{TPOT_{C0}}{TPOT_{C2}}
]

[
E2EReduction=1-rac{GenerationTime_{C2}}{GenerationTime_{C0}}
]

[
RemoteReduction=1-rac{RemotePairs_{C2}}{RemotePairs_{C0}}
]

Also report:

[
FetchChange=
rac{Fetches_{C2}-Fetches_{C0}}{Fetches_{C0}}.
]

If C2 changes physical H2D materially, do not attribute the full E2E gain to NVLink communication.

C1 is the global-assignment ablation between Random and C2.

## 7. Relationship analysis

Across Stage-B cells/repeats plot:

1. RemotePairFraction vs TPOT;
2. RemotePairFraction vs generation time;
3. physical H2D GiB vs generation time.

Descriptive regression:

[
T_{gen}
=
b_0+b_1 RemotePairs+b_2 H2DBytes+b_3 ControllerTime+b_4 RankCV.
]

This regression is descriptive only. Stage A supplies the isolated communication evidence.

## 8. Required figures

1. Random vs Hungarian-current vs Hungarian-same+path: TPOT and generation time for R4/R8.
2. RemotePairFraction vs MoE layer latency from Stage A.
3. RemotePairFraction vs TPOT from Stage B.
4. Random vs C2 system metrics: H2D GiB, peer payload GiB, controller time and rank CV. Do not stack overlapping times.

## 9. Acceptance criteria

A C-policy E2E claim requires:

1. C2 median TPOT lower than Random in both R4 and R8 primary anchors;
2. all five repeat ranges reported;
3. C2 RemotePairFraction lower in both anchors;
4. runtime/accounting gates pass;
5. actual expert H2D equals logical fetch bytes in posthoc profiles;
6. the result is reproduced without profiler instrumentation.

If communication falls but TPOT/E2E does not, publish the negative result. Do not retune after timing is visible.

## 10. Validation gates

Before measurement:

- reuse R1/R4/R8 native parity tests;
- direct slot views must show zero full-expert D2D copies on cache hit;
- actual H2D bytes = logical fetch bytes;
- fixed-seed policy decisions/tokens deterministic;
- local/remote counters independently recomputed from origin/destination maps;
- Stage-A timed ranges assert zero expert H2D.

## 11. Scope

This study supports H100 NVSwitch physical-runtime claims only.

It does not establish:

- PCIe-only GPU behavior;
- remote NUMA behavior;
- production serving throughput;
- long-horizon task accuracy.

## 12. Stop condition

After Stage A, uninstrumented Stage B and the two posthoc profiles, stop for owner review.

Do not automatically retune substitution, eviction or affinity coefficients, add replication/migration, or change hard quotas.
