# PLAN — Physical GPU offloading E2E / TPOT

Date: 2026-10-03
Status: prospective, owner-authorized.

## 1. Questions

1. Does CA's lower peer traffic translate into lower **E2E** and **TPOT**?
2. Is the CA gain larger in **Env 2** than in **Env 1**?
3. Does CA-rep become worthwhile in Env 2 despite its extra H2D/cache cost?
4. Does substitution reduce miss pressure without invalidating the placement
   trend?

This is a physical execution study, not a communication-only replay.

---

## 2. What "actual GPU offloading" means

The timed path must execute the real Qwen3-30B-A3B model/runtime.

- expert weights live in the CPU expert pool unless resident in a GPU cache;
- a real cache miss performs a full expert CPU->GPU H2D transfer to the
  policy-selected rank;
- GPU expert cache capacity is enforced from the selected global cache ratio;
- routed token activations use real inter-rank communication;
- the selected GPU expert kernel actually executes;
- substitution ON actually executes the selected substitute expert;
- evictions remove physical GPU expert copies and later reloads incur real H2D.

A run that merely sleeps, replays byte counts, or times synthetic memcpy/NCCL
is not accepted as an E2E result.

---

## 3. Environments

### Env 1 — NVSwitch

Timed process:
- `NCCL_CUMEM_ENABLE=0`;
- P2P enabled;
- no NCCL debug logging.

An untimed preflight must verify the expected P2P/IPC path and no NET path.

### Env 2 — P2P disabled

Timed process:
- `NCCL_CUMEM_ENABLE=0`;
- `NCCL_P2P_DISABLE=1`;
- `NCCL_IB_DISABLE=1`;
- no NCCL debug logging.

An untimed preflight must verify SHM/direct/direct with no P2P and no NET path.

Env 2 is the project's controlled expensive-communication condition; do not
claim it is equivalent to a particular physical PCIe topology.

---

## 4. Frozen representative cells

Use the exact request manifests and model checkpoint from the completed
BR/CA/CA-rep packet. Decode length is 256.

### P — placement cell

- dataset: ShareGPT;
- R=8;
- local batch=16 (global 128);
- cache=40% global slots;
- eviction=Gate;
- substitution=ON;
- policies={BR, CA, CA-rep}.

Rationale: general conversational workload, R8, moderate physical cost, and the
CPU study shows clear CA communication headroom.

### R — replication-stress cell

- dataset: MATH;
- R=4;
- local batch=64 (global 256);
- cache=30% global slots;
- eviction=Gate;
- substitution=ON;
- policies={BR, CA, CA-rep}.

Rationale: long decode / large local batch gives CA-rep one of the clearest
communication-saving opportunities while keeping the cache constrained.

If R4/B64 fails a **resource preflight** (OOM or safety guard) before any timing,
the only allowed fallback is R4/B32 with all other axes unchanged. Do not
fallback because of an unfavorable performance result.

### E — exact-only placement control

Same as P, except:
- substitution=OFF;
- policies={BR, CA} only.

Purpose: establish the placement timing effect without substitution-induced
trajectory coupling.

---

## 5. Eviction and substitution

### Primary eviction = Gate

Use the exact GateScore semantics from the completed CPU study:
- W=128;
- normal active/pinned safety only;
- no replica protection;
- no Coverage eviction.

Gate is frozen for physical timing because the CPU sweep already compared
LRU/Gate and Gate reduces miss/reload pressure in many constrained cells.
Repeating the full LRU axis physically would multiply cost without answering
the Env question.

### Substitution = ON in P and R

Use the already verified:
- FineWeb-Edu 400x128 SERE Frobenius similarity;
- gate protect threshold 0.20;
- similarity threshold 0.65.

E is the matched substitution-OFF control.

### Conditional LRU check

Only if BR->CA timing in P is reversed relative to its clean counter pass by
>=5% in **both** Env 1 and Env 2 may one diagnostic P/Env2/BR-vs-CA LRU pair be
run. This diagnostic cannot replace the primary Gate result.

---

## 6. Remove controller overhead from the timed region

The CPU experiment measured policy headroom, so this physical packet must first
measure the execution consequence, not Python/Hungarian/oracle cost.

For every (cell, policy):

### Phase A — PLAN (untimed, instrumented)

Run the policy once to produce a frozen per-event action schedule:
- primary placements;
- replica admissions;
- substitution mappings;
- evictions/fetches;
- expected route/miss sets;
- final token IDs, route hash, action hash and cache-state hash.

CA assignment and CA-rep oracle work happen here, outside timing.

### Phase B — COMPILE (untimed)

Fresh process, same Env and frozen schedule:
- initialize NCCL;
- load model and CPU expert pool;
- allocate caches/streams;
- exercise prefill, H2D, communication, substitution and expert kernels;
- populate a dedicated TorchInductor/Triton cache directory;
- no scientific timing is reported.

The compile directory is keyed by Env/cell/policy.

### Phase C — MEASURE (clean timed process)

Fresh process using the compiled cache.

Before timing:
1. load compiled artifacts;
2. initialize NCCL;
3. execute one untimed warmup;
4. reset expert-cache state, allocator-visible policy state, RNG and request
   position to the frozen initial state;
5. synchronize all target GPUs.

Inside the timed region:
- no per-token print;
- no per-layer CUDA event;
- no Nsight;
- no NCCL debug;
- no hit/miss counter updates requiring Python/device synchronization;
- no route dumps;
- no controller optimization;
- no compilation.

Only boundary timing is permitted.

After the timed region:
- synchronize;
- save output-token hash and final-state hash;
- verify them against PLAN receipts.

If a route/action mismatch makes the frozen schedule invalid, the cell fails
validation; do not silently recompute the controller inside timing.

### Phase D — COUNTERS (untimed, instrumented)

Run the same frozen schedule once outside timing to collect:
- exact global/local hit;
- resident substitute hit;
- effective hit;
- residual miss;
- H2D bytes and fetches;
- reload bytes/count;
- peer bytes;
- local-service fraction;
- evictions/turnover;
- replica admissions/reuse/victim reloads.

COUNTERS explains the timing result but is never used as the timing sample.

---

## 7. Timing definitions

Primary:

### E2E
Wall-clock from immediately before prefill to completion of the final decode
token, with all target GPUs synchronized at the outer boundaries.

### TPOT
Decode GPU elapsed time / 256 output tokens.

Use only a constant number of boundary CUDA events; never one event per token.
Also report decode wall time as a consistency check.

Do not include:
- process startup;
- model load;
- NCCL initialization;
- PLAN;
- compile;
- compile-cache load warmup;
- COUNTERS/report generation.

Do include during E2E:
- real prefill;
- real CPU expert H2D misses;
- real inter-rank activation communication;
- expert execution;
- cache eviction/reload;
- substitution execution when ON;
- token selection.

---

## 8. Repetitions and order

For each primary timed cell:
- 1 untimed warmup after compile-cache load;
- 3 clean timed repetitions;
- reset identical initial cache/policy/request state before each repetition.

Counterbalance policy order across repeats:
- repeat 1: BR -> CA -> CA-rep;
- repeat 2: CA-rep -> CA -> BR;
- repeat 3: CA -> BR -> CA-rep.

For E, use BR/CA and reverse order on alternating repetitions.

Env 1 and Env 2 order must also be counterbalanced across cells.

Report median and full range. If the paired E2E delta of the main comparison is
smaller than the within-policy range in either environment, run exactly two
additional repeats for that implicated pair only. Maximum 5 timed repeats per
policy; no open-ended repetition.

---

## 9. Primary matrix

| Cell | Env | Sub | Evict | Policies |
|---|---|---|---|---|
| P | Env 1 | ON | Gate | BR, CA, CA-rep |
| P | Env 2 | ON | Gate | BR, CA, CA-rep |
| R | Env 1 | ON | Gate | BR, CA, CA-rep |
| R | Env 2 | ON | Gate | BR, CA, CA-rep |
| E | Env 1 | OFF | Gate | BR, CA |
| E | Env 2 | OFF | Gate | BR, CA |

Total primary policy/environment cells = 16.
At 3 repeats = 48 timed generations before any targeted noise repeat.

No full cache/batch/R/eviction sweep is authorized.

---

## 10. Required comparisons

### Placement

For P and E:

```
gain_CA = (T_BR - T_CA) / T_BR
```

Report E2E and TPOT separately.

Main hypothesis:
```
gain_CA(Env2) > gain_CA(Env1)
```

### Replication

For P and R:

```
gain_rep = (T_CA - T_CA-rep) / T_CA
```

The question is whether extra H2D that looked unattractive in CPU bytes becomes
worthwhile under Env 2's more expensive peer communication.

### Mechanism check

Each timing comparison must be accompanied by its separate COUNTERS pass:
- peer reduction should agree in direction with the frozen CPU headroom;
- H2D/reload deltas must explain any reversal;
- exact/effective hit and residual miss must be shown next to timing.

Do not infer causality from timing alone if the counter pass disagrees.

---

## 11. Validation gates

Before accepting timing:
- Env path preflight passes;
- real expert H2D bytes > 0;
- resident expert copies never exceed configured cache slots;
- CPU expert pool is not silently preloaded into GPU memory;
- PLAN/MEASURE output token and final-state hashes match per policy;
- no compile occurs after the MEASURE timer starts;
- no profiler or per-event logger is active;
- COUNTERS is a separate process/pass;
- all target GPUs start below the packet's thermal/load safety thresholds;
- resident idle-load workers are stopped and GPU memory is released.

After completion, restore the owner's resident model workers only after all
scientific processes have exited.

---

## 12. Scope

No accuracy claim, controller-overhead claim, new threshold tuning, new
substitution calibration, broad physical sweep, Nsight run, or PCIe-equivalence
claim.

Commit PLAN/COMPILE/MEASURE/COUNTERS receipts separately. Stop for owner review
after the bounded matrix.
