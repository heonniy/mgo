# PLAN — BR / CA / CA-rep CPU headroom

Date: 2026-10-03
Status: prospective, owner-authorized.
Priority: supersedes further dynamic-replica-refresh exploration after its
current safe checkpoint.

## 1. Research question

Main question:

> In multi-GPU offloading, does expensive inter-GPU communication make the
> destination rank of a missed expert a first-class admission decision?

The later physical comparison will use:
- **Env 1** = NVSwitch.
- **Env 2** = P2P disabled.

This packet does not run transport timing. It characterizes the physical
resource headroom of **BR / CA / CA-rep** first.

The intended later hypotheses are:
- Env 1: BR may already be sufficient because remote communication is cheap.
- Env 2: CA should become more valuable.
- If communication pressure is sufficiently strong, CA-rep may improve beyond
  CA for persistently rank-local experts.

---

## 2. Dataset and decode-heavy workload

Use **MATH (Hendrycks et al.)** only.

### 2.1 Workload set

Use **512 examples from the MATH test split**.

Selection:
- deterministic seed 42;
- stratify by MATH subject/category and difficulty level as evenly as possible;
- freeze exact example IDs, problem hashes and dataset revision in a manifest;
- no sample is shared with substitution calibration.

Prompt:
```
Solve the following problem step by step. Keep the reasoning explicit and put
the final answer at the end.
```

Generation:
- greedy / deterministic;
- max input tokens = 512;
- exactly 64 new tokens per request
  (`min_new_tokens=max_new_tokens=64`);
- accuracy is not evaluated.

The fixed 64-token continuation makes this a controlled decode-heavy systems
workload and keeps all requests active for the same decode horizon.

### 2.2 Master trace

Make **one exact-model master capture** using all 8 GPUs:
- logical R=8;
- local batch=64;
- global requests=512;
- one prefill plus exactly 64 decode steps;
- substitution OFF during capture;
- no timing claim.

The master trace must retain enough per-request data to re-pack workloads
offline:
- per request/token/layer top-k experts;
- selected routing weights;
- full router probabilities in a compact lossless-enough representation for
  deterministic W128 GateHistory replay; prefer float32 if required for exact
  victim parity;
- token IDs / active-mask;
- per-request ordering and hashes.

Do not retain giant profiler traces.

### 2.3 Logical packing

All CPU cells are derived from this one master trace.

For each (R,B), use the first `N=R*B` requests from the frozen stratified
master order and assign origins round-robin so every rank has exactly B
requests.

This gives:

| R | local B | global requests |
|---:|---:|---:|
| 4 | 8 | 32 |
| 4 | 16 | 64 |
| 4 | 32 | 128 |
| 4 | 64 | 256 |
| 8 | 8 | 64 |
| 8 | 16 | 128 |
| 8 | 32 | 256 |
| 8 | 64 | 512 |

Therefore same-global comparisons are available:
- R4/B16 vs R8/B8 (64);
- R4/B32 vs R8/B16 (128);
- R4/B64 vs R8/B32 (256).

This is a frozen-routing headroom study: changing logical R/B changes request
aggregation/origin ranks/cache partitioning, not the already generated model
tokens.

---

## 3. Substitution calibration

The old `similarity.npy` is not used as the primary calibration for this
packet.

Use a disjoint **128-example MATH train calibration set**, stratified by the
same subject/difficulty metadata, deterministic seed 43.

Run calibration in the same 8-GPU model session before the workload capture:
- max input 512;
- 32 fixed decode tokens;
- exact model;
- substitution disabled while collecting calibration evidence.

### 3.1 Similarity definition

Use **co-routed expert output similarity**, avoiding all-expert brute-force
evaluation.

For each token and MoE layer, the top-k routed experts see the same hidden
state. For every co-routed expert pair (i,j), accumulate:

```
S_l(i,j) = mean cosine( expert_i_output(h), expert_j_output(h) )
```

over shared calibration tokens where both experts were routed.

Also record co-observation counts.

A pair is eligible for substitution only when:
- cosine similarity >= 0.65; and
- co-observation count >= 16.

Gate protection remains frozen at 0.20.

Unsupported pairs are never used as substitutes.

### 3.2 Calibration sufficiency gate

For every (layer,expert), require at least 4 eligible/observed candidate
neighbors with count >=16 for at least 95% of layer-expert cells.

Also compute split-half top-4-neighbor overlap as a stability diagnostic.

If the coverage gate fails at 128 calibration examples, extend exactly once
with 128 additional disjoint MATH-train examples (256 total). No further
extension.

If the 256-example gate still fails:
- mark substitution calibration `INSUFFICIENT`;
- run the full substitution-OFF matrix;
- skip substitution-ON cells rather than inventing unsupported similarities.

No accuracy/quality validation is part of this packet.

---

## 4. Global cache budget and eviction

R in {4,8}.

Cache ratio is a **global-slot ratio**, fixed independently of R:

```
cache = {30%,40%,50%,60%}
global_slots = floor(48 * 128 * cache_ratio)
```

Split global slots across R ranks as evenly as possible. R=8 does not silently
receive twice the total expert-cache budget of R=4.

Eviction:
- LRU;
- Gate-score (W=128).

No Coverage eviction in this packet.

Substitution:
- OFF;
- ON using the new MATH calibration and frozen thresholds:
  gate protect=0.20, similarity=0.65.

---

## 5. Hit / miss metrics

Report both route-count weighted and gate-mass weighted forms.

Before miss admission at every event:

### Exact global hit
The requested exact expert has at least one resident copy on any rank.

### Exact local hit
The requested exact expert already has a copy on the request's origin rank.

### Substitute hit
The exact expert is globally missing, but the frozen substitution policy maps
it to an already-resident eligible substitute, so no exact H2D fetch is needed.

### Effective hit
```
effective_hit = exact_global_hit + substitute_hit
```
with disjoint counting.

### Residual miss
After optional substitution, an exact expert must still be fetched from CPU.

Also report:
- effective local hit: local exact service or local resident substitute service;
- global exact-hit rate;
- local exact-hit rate;
- substitute-hit rate;
- effective-hit rate;
- residual-miss rate;
- unique expert-event miss rate;
- reload rate;
- evictions / global slot / decode step (cache turnover);
- mean unique resident experts;
- per-rank fetch counts and imbalance.

These metrics are required before interpreting BR/CA/CA-rep.

---

## 6. BR

**BR = Balanced Random.**

For each layer-event, collect residual exact-miss experts after substitution.

Create balanced rank quotas whose counts differ by at most one.

Assign miss experts randomly to rank quota slots with deterministic seed 42.

Properties:
- every expert gets exactly one primary copy;
- rank fetch-count balance is enforced;
- current rank demand is NOT used in assignment;
- same cache/eviction/substitution state semantics as other policies.

Full grid uses seed 42.

Randomness audit only:
run seeds {7,42,99} on the following 8 substitution-OFF/Gate anchors:
- R={4,8};
- B={8,64};
- cache={30%,60%}.

No other seed sweep.

---

## 7. CA

**CA = Comm-aware Balanced current-demand oracle.**

Use exactly the same balanced rank quotas as BR.

For each residual miss expert e and rank r, compute current-event demand
`d[e,r]` from effective routes.

Solve the exact balanced assignment:

```
maximize sum_e,r d[e,r] * x[e,r]
subject to:
  each miss expert assigned to exactly one rank
  rank assignment counts equal the BR quotas
```

Equivalent exact peer-byte minimization may be used when dispatch/combine
interactions are recomputed exactly.

This is a current-event oracle:
- no future routing knowledge;
- no controller-overhead measurement;
- no quota relaxation;
- no migration of already-resident experts.

BR vs CA therefore isolates **where the same balanced set of mandatory H2D
fetches is placed**.

---

## 8. CA-rep

**CA-rep = CA plus selective future-popularity oracle replication.**

First run the mandatory CA primary placement.

For each newly admitted residual-miss expert e:
- consider at most ONE extra replica;
- candidate ranks exclude the primary;
- choose the non-primary rank with the largest cumulative future raw demand for
  e over the remaining same-layer decode opportunities in the frozen 64-step
  trace.

Define optimistic future communication value:

```
V(e,r) = cumulative peer activation bytes that a persistent local copy on r
         could avoid over the remaining same-layer decode opportunities.
```

Admission rule:
```
replicate e on best r only if V(e,r) >= one expert payload = 9 MiB.
```

This is deliberately per-expert, not a global replica-ratio policy.

Replica semantics:
- one extra copy maximum per miss expert;
- replica costs a normal 9-MiB H2D fetch;
- no replica protection beyond normal same-event active protection;
- after admission, it is an ordinary cache entry;
- LRU/Gate may evict it naturally;
- a replica may evict another legal cache entry, causing later real reloads;
- all those H2D/reload effects are counted;
- no migration/refresh of an already-resident replica.

The oracle sees future **raw demand only**, not future cache evictions or future
substitution outcomes. Actual replay determines realized benefit.

For diagnosis, report `V / 9MiB` candidate fractions in bins:
`<.25, .25-.5, .5-1, 1-2, >=2`.
Do not add extra CA-rep policies for those bins.

---

## 9. CPU matrix

Base system configurations:

```
R:            2  = {4,8}
local batch:  4  = {8,16,32,64}
cache:        4  = {30,40,50,60%}
eviction:     2  = {LRU,Gate}
substitution: 2  = {OFF,ON}
--------------------------------
128 base configurations
```

For every valid base configuration run:
- BR;
- CA;
- CA-rep.

Main replay count:
```
128 * 3 = 384
```

Add 16 extra BR seed-audit replays (8 anchors x two non-default seeds).

Maximum CPU policy replays:
```
400
```

If substitution calibration is insufficient, skip the 64 substitution-ON base
configurations and record the reduced matrix explicitly.

One CPU process per replay; parallelize conservatively only if aggregate host
RAM remains under the existing server guard. No GPU is used after the single
calibration/master-capture session.

---

## 10. Required BR / CA / CA-rep headroom outputs

For every base configuration report:

### BR -> CA
- peer-byte reduction;
- remote token-rank-pair reduction;
- local-service increase;
- H2D delta;
- reload delta;
- rank-fetch imbalance delta.

This is the **single-copy placement headroom**.

### CA -> CA-rep
- peer-byte reduction;
- extra H2D;
- replica admissions;
- replica reuse before eviction;
- victim-induced reloads;
- local-service increase;
- duplicate occupancy.

This is the **selective replication headroom**.

### R / batch trends
Primary plots:
1. BR vs CA peer reduction over R x B;
2. CA vs CA-rep H2D-peer tradeoff over R x B;
3. exact/substitute/effective hit and residual miss over cache x batch;
4. cache turnover/reload rate;
5. R4 vs R8 at equal global request count;
6. substitution OFF vs ON;
7. LRU vs Gate.

No artificial weighted "winner" is required.

---

## 11. Predeclared interpretation labels

### CA_HEADROOM
CA reduces peer bytes >=10% versus BR while:
- H2D bytes are within +/-2%; and
- maximum/minimum rank mandatory-fetch count differs by no more than BR +1.

### CA_STRONG_HEADROOM
Same constraints, peer reduction >=25%.

### CA_REP_HEADROOM
CA-rep reduces peer bytes >=20% versus CA while H2D increases <=15%.

### CA_REP_TRADEOFF
CA-rep reduces peer bytes >=20% but H2D increases >15%; report the raw Pareto
movement without calling it a system win.

### SUBSTITUTION_RELIEVES_MISS_PRESSURE
Substitution ON reduces residual-miss rate or reload bytes >=20% versus matched
OFF while reporting the substitute route/gate-mass fraction.

### HIGH_MISS_PRESSURE
Residual-miss rate >=30% OR decode cache-turnover >=1 global cache per decode
step. This is the regime where substitution/system admission pressure is
explicitly flagged.

Labels are descriptive CPU headroom labels only, not Env 1/Env 2 latency claims.

---

## 12. Resource-efficient GPU execution

Use all 8 GPUs for one bounded model session only:

1. load the exact Qwen3-30B-A3B model/runtime once;
2. collect 128-sample MATH-train substitution calibration;
3. extend to 256 only if the frozen calibration coverage gate fails;
4. capture the 512-request MATH-test master trace with 64 fixed decode tokens;
5. hash/validate artifacts;
6. unload and return to CPU-only replay.

No separate GPU capture for each R, batch, cache, eviction, substitution or
policy.

This is the main cost-saving design.

Preserve the server's existing ownership/resource guards. Do not kill unrelated
jobs.

---

## 13. Scope boundary

This packet does NOT:
- measure Env 1 or Env 2 E2E/TPOT;
- run NCCL timing;
- evaluate accuracy;
- tune substitution thresholds;
- tune an online CA controller;
- use a replica ratio rho;
- protect replicas;
- run B128;
- run Coverage eviction.

After the CPU headroom result, stop for owner review. The next packet, only if
authorized, will pick a minimal representative subset for Env 1 / Env 2 E2E.
