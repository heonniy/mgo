# PLAN — BR / CA / CA-rep CPU headroom

Date: 2026-10-03
Status: prospective, owner-authorized.

## 1. Research question

> In multi-GPU offloading, does expensive inter-GPU communication make the
> destination rank of a missed expert a first-class admission decision?

Later physical validation will use:
- **Env 1** = NVSwitch.
- **Env 2** = P2P disabled.

This packet only measures BR / CA / CA-rep resource headroom using frozen
routing traces. No Env timing or accuracy evaluation is included.

---

## 2. Workload datasets

Use two distinct decode-heavy workloads so conclusions are not tied only to
mathematical reasoning.

### 2.1 MATH

Purpose: reasoning-heavy workload with structured multi-step generations.

- source: MATH test split;
- samples: 512;
- deterministic seed 42;
- stratify by subject/category and difficulty as evenly as available;
- max input tokens: 512;
- greedy/deterministic generation;
- exactly 256 new tokens per request
  (min_new_tokens=max_new_tokens=256);
- the first 64 decode steps are retained as an exact prefix horizon for a
  controlled 64-vs-256 comparison;
- accuracy is ignored.

### 2.2 ShareGPT

Purpose: general conversational / everyday serving workload with naturally
long assistant responses. This is the second, non-reasoning workload.

Use a frozen ShareGPT V3 cleaned conversation source and record the exact
dataset path/revision in the manifest.

Select 512 human->assistant turns:
- deterministic seed 44;
- tokenize with the target Qwen tokenizer before filtering;
- user-prompt length: 32--512 tokens;
- reference assistant response length: at least 128 tokens;
- discard malformed/missing turns and duplicates;
- preserve the selected user prompt as the model request;
- max input tokens: 512;
- greedy/deterministic generation;
- exactly 256 new tokens per request;
- accuracy/reference matching is ignored.

The response-length filter selects naturally decode-heavy conversational
requests. A single 256-token capture provides two exact horizons from the same
generation: decode64 = the first 64 decode steps, and decode256 = the full
trace. No separate 64-token GPU capture is allowed.

### 2.3 Master traces

Use all 8 GPUs in **one model-loading session**.

Capture two exact-model master traces sequentially:
1. MATH: 512 requests;
2. ShareGPT: 512 requests.

For each master capture:
- logical R=8;
- local batch=64;
- global requests=512;
- substitution OFF while capturing;
- one prefill + exactly 256 decode steps;
- expose two trace horizons from the same raw capture:
  - **decode64** = first 64 decode steps;
  - **decode256** = all 256 decode steps;
- no timing claim.

Retain per request/token/layer:
- top-k expert IDs;
- selected routing weights;
- compact full-router information sufficient to replay W128 GateScore exactly;
- token IDs / active mask;
- request ordering and hashes.

Do not store profiler traces.

All R/batch CPU workloads are repacked from the corresponding dataset's one
master trace. The 64-step workload must be obtained by truncating the same
256-step trace, never by a second generation run.

For each dataset:

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

Use the first N requests from the frozen dataset order and assign origin ranks
round-robin so every rank receives exactly B requests.

Same-global comparisons:
- R4/B16 vs R8/B8 = 64 requests;
- R4/B32 vs R8/B16 = 128;
- R4/B64 vs R8/B32 = 256.

---


### 2.4 Decode-horizon trace audit

Immediately after each 256-step master capture, produce a lightweight raw-routing
comparison for **decode64 vs decode256** from the same requests and same generated
trajectory prefix. This is a trace characterization only; it does not change the
BR / CA / CA-rep CPU matrix in this amendment.

Required per-dataset horizon statistics:
- total routed expert routes and gate mass;
- active unique experts per layer/step;
- rank-local expert demand p50/p90/p99/max;
- top-10% rank-local demand share;
- same-layer expert/rank recurrence over future decode steps;
- fraction of expert/rank pairs seen in decode64 that recur after step 64;
- hot-set overlap between steps 1--64 and 65--256;
- per-layer demand-skew summary.

The purpose is to answer cheaply whether a 64-token decode trace materially
under-represents longer-horizon expert popularity/reuse. Store compact CSV/JSON
summaries only. No extra GPU execution is allowed for this comparison.

**Important scope:** this amendment changes only the 8-GPU trace collection and
adds the raw 64-vs-256 horizon audit. The existing CPU replay grid/budget is not
automatically doubled to both horizons. Because the 256-step master trace stores
both horizons, a later owner decision can run the BR / CA / CA-rep CPU matrix at
64, 256, or a selected subset without recapturing the model.

---

## 3. Substitution calibration — SERE-style

Do **not** calibrate expert similarity on MATH or ShareGPT.

The primary calibration must reproduce the SERE-style calibration already used
in this project:

- dataset: **FineWeb-Edu**;
- data volume: **400 sequences x 128 tokens** = 51,200 tokens;
- deterministic seed 42 for the frozen subset;
- similarity metric: **SERE Frobenius output similarity**;
- layer-wise expert similarity matrices normalized exactly as the existing
  SERE-compatible implementation;
- target model: the same Qwen3-30B-A3B checkpoint used for workload capture.

Substitution thresholds remain:
- gate protection = 0.20;
- similarity threshold = 0.65.

### 3.1 Reuse before recalibration

First audit the existing similarity artifact used by the project.

If provenance confirms that it was produced from:
- this exact model checkpoint;
- FineWeb-Edu 400 x 128;
- the SERE Frobenius implementation;

then hash-pin and **reuse it**. Do not recalibrate.

If provenance is missing or mismatched, run exactly one FineWeb-Edu SERE
calibration in the same 8-GPU session before the two workload captures.

Do not replace SERE calibration with co-routed cosine or a workload-specific
similarity matrix.

No accuracy validation is required in this packet.

---

## 4. Cache / eviction / substitution grid

Ranks:
```
R={4,8}
```

Local batches:
```
B={8,16,32,64}
```

Cache ratio is a **global-slot ratio**, independent of R:
```
cache={30%,40%,50%,60%}
global_slots=floor(48*128*cache_ratio)
```
Split those slots across ranks as evenly as possible.

Eviction:
- LRU;
- Gate-score, W=128.

Substitution:
- OFF;
- ON using the SERE-style FineWeb-Edu similarity and frozen thresholds.

No Coverage eviction.

---

## 5. Required hit / miss accounting

Report route-count weighted and gate-mass weighted metrics.

Before admission at each event:

### Exact global hit
Requested exact expert has any resident copy.

### Exact local hit
Requested exact expert is already resident on the origin rank.

### Substitute hit
Exact expert is globally missing, but the frozen substitution policy can route
it to a resident eligible substitute, avoiding an exact CPU fetch.

### Effective hit
```
effective_hit = exact_global_hit + substitute_hit
```
with disjoint counting.

### Residual miss
After optional substitution, the exact expert must still be fetched.

Also report:
- effective local hit;
- unique expert-event miss rate;
- reload rate;
- cache turnover = evictions / global_slots / decode_step;
- mean unique resident experts;
- per-rank mandatory fetch counts and imbalance.

These metrics are required for every dataset / R / B / cache / eviction /
substitution configuration before interpreting policy headroom.

---

## 6. BR

**BR = Balanced Random.**

For every layer-event after substitution, collect residual exact-miss experts.

Create rank quotas whose counts differ by at most one, then randomly assign
experts to quota slots with deterministic seed 42.

Properties:
- one primary copy per residual miss;
- balanced mandatory fetch count;
- no rank-demand information in the assignment.

Randomness audit only: use seeds 7 and 99 in addition to seed42 for 8 total
extreme anchors across the two datasets. No full seed sweep.

---

## 7. CA

**CA = Comm-aware Balanced current-demand oracle.**

Use exactly the same balanced rank quotas as BR.

For residual miss expert e and rank r, compute current-event effective-route
rank demand d[e,r].

Solve the exact balanced assignment maximizing local demand (equivalently,
minimizing exact peer traffic under the frozen event accounting) while keeping
the BR quota fixed.

CA:
- uses current event demand only;
- has no future knowledge;
- does not migrate already-resident experts;
- does not relax rank quotas;
- has no controller-overhead measurement in this packet.

BR vs CA isolates the value of **where** the same mandatory fetches are placed.

---

## 8. CA-rep

**CA-rep = CA plus selective future-popularity oracle replication.**

Mandatory primary placement is CA.

For each newly admitted residual-miss expert:
- consider at most one additional replica;
- candidate ranks exclude its CA primary rank;
- use the remaining same-layer decode demand in the frozen trace to find the
  best persistent non-primary rank.

For the one chosen rank compute:
```
V(e,r) = optimistic cumulative peer bytes a persistent local copy could avoid
         over remaining same-layer decode opportunities.
```

Admission rule:
```
replicate only when V(e,r) >= 9 MiB
```
(one expert payload).

This is a per-expert decision, not a replica-ratio policy.

Replica semantics:
- costs one normal 9-MiB H2D;
- no protection except same-event active protection;
- ordinary LRU/Gate eviction afterwards;
- can create victim-induced future reloads;
- no migration or refresh.

The oracle sees future raw demand only, not future cache evictions or future
substitution decisions. Actual replay determines realized costs.

Also report V/9MiB bins:
<.25, .25-.5, .5-1, 1-2, >=2.

---

## 9. CPU matrix and cost control

Per dataset, base system configurations:
```
R:            2
B:            4
cache:        4
eviction:     2
substitution: 2
----------------
128 base configs
```

Policies:
- BR;
- CA;
- CA-rep.

Per dataset:
```
128*3 = 384 main replays
```

Two datasets:
```
768 main CPU replays
```

Add at most 16 total extra BR replays for the frozen seed audit across both
datasets.

Hard maximum:
```
784 CPU policy replays
```

### Parallel execution

CPU replay is single-threaded per cell. Run at most 16 cells concurrently,
subject to:
- aggregate replay RSS <=32 GiB;
- host available memory >=512 GiB before launching a wave;
- BLAS/OMP=1 per process;
- CUDA hidden during CPU replay.

Use deterministic cell checkpoints so completed cells are never rerun after an
interruption.

The expensive model is loaded only once for:
- optional one-time SERE calibration if reuse is impossible;
- MATH 256-decode master capture;
- ShareGPT 256-decode master capture.

There is no separate decode64 capture; decode64 is the first 64 steps of each
decode256 master trace.

No GPU run is repeated for R/cache/eviction/substitution/policy combinations.

---

## 10. Required BR / CA / CA-rep outputs

For every base configuration:

### BR -> CA placement headroom
- peer-byte reduction;
- remote token-rank-pair reduction;
- local-service increase;
- H2D delta;
- reload delta;
- rank-fetch imbalance delta.

### CA -> CA-rep replication headroom
- peer-byte reduction;
- extra H2D;
- replica admissions;
- replica reuse before eviction;
- victim-induced reloads;
- local-service increase;
- duplicate occupancy.

### Required trend views
1. BR vs CA peer reduction over R x B;
2. CA vs CA-rep H2D-peer movement over R x B;
3. exact/substitute/effective hit and residual miss over cache x batch;
4. cache turnover / reload rate;
5. R4 vs R8 at equal global request count;
6. MATH vs ShareGPT;
7. substitution OFF vs ON;
8. LRU vs Gate.

Do not force a single weighted winner.

---

## 11. Descriptive labels

### CA_HEADROOM
CA reduces peer bytes >=10% versus BR while:
- H2D within +/-2%;
- mandatory rank-fetch imbalance no worse than BR +1 expert.

### CA_STRONG_HEADROOM
Same constraints, peer reduction >=25%.

### CA_REP_HEADROOM
CA-rep reduces peer bytes >=20% versus CA while H2D increases <=15%.

### CA_REP_TRADEOFF
Peer reduction >=20% but H2D increase >15%.

### SUBSTITUTION_RELIEVES_MISS_PRESSURE
Substitution ON reduces residual-miss rate or reload bytes >=20% versus matched
OFF, with substitute route/gate-mass fraction reported.

### HIGH_MISS_PRESSURE
Residual-miss rate >=30% OR decode cache-turnover >=1 global cache per decode
step.

These are CPU resource-headroom labels, not Env 1/Env 2 latency claims.

---

## 12. Scope boundary

No:
- Env 1 / Env 2 E2E or TPOT;
- NCCL timing;
- accuracy evaluation;
- substitution threshold tuning;
- workload-specific similarity calibration;
- online CA controller timing;
- replica ratio rho;
- replica protection;
- B128;
- Coverage eviction.

After CPU results, stop for owner review. A later packet may select only a
minimal representative subset for Env 1 / Env 2 E2E.
