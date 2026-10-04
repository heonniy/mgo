# MILESTONES — implementation sequence and gates

Do not combine milestones. Each stage must pass its own parity/invariant gate
before the next stage begins.

## M0 — Branch freeze and scope cleanup

### Work
- branch: `refactoring`;
- freeze exact baseline commit and route traces;
- mark replication code/config as inactive for this path;
- substitution OFF;
- add one unified experiment config.

### Tests
- import/static tests;
- baseline BR/CA/LA CPU replay;
- physical short decode hash parity.

### Exit gate
Frozen baseline receipt with:
- source SHA;
- route/token hashes;
- cache hit/miss/H2D counters.

---

## M1 — Baseline phase instrumentation

### Work
Add NVTX/CUDA timing without changing scheduling.

Ranges:
- metadata;
- current controller;
- H2D;
- forward transport;
- expert compute;
- return transport;
- combine.

### Tests
Instrumentation ON/OFF token parity.

### Exit gate
One B128 short run where phase union reconciles with measured MoE span within a
documented tolerance.

---

## M2 — EdgeMoE-style next-layer calibration

### Work
Build calibration tool:
- extract token-level top-k routes;
- build `T_l[E,E]`;
- freeze calibration/eval split;
- serialize table with hashes.

### Tests
- rows normalize correctly;
- held-out requests never enter calibration;
- deterministic table generation.

### Output
- transition matrices;
- Precision/Recall@P;
- actual-miss Recall@P;
- per-rank predicted-demand error.

### Exit gate
Prediction quality report for B128/B256 traces and oracle gap.

---

## M3 — CPU prefetch budget / placement characterization

### Work
Replay P in {0,1,2,4,8} with:
- BR-prefetch;
- CA-prefetch;
- LA-prefetch.

Model dedicated C+P capacity and promotion victim effects.

### Required outputs
- useful/wasted predictions;
- promotion count;
- promotion-induced evictions;
- future reload delta;
- per-rank prefetch count;
- predicted vs actual owner regret.

### Exit gate
Pick only a small physical sweep set, e.g. P={1,2,4} around the CPU knee.
Do not declare a winner yet.

---

## M4 — C+P slot arena and promotion state machine

### Work
Implement one physical arena per rank:
- C MAIN roles;
- P PREFETCH roles;
- PREFETCH EMPTY/QUEUED/INFLIGHT/READY states;
- deterministic global reservation shadow;
- rank-local CUDA readiness.

Implement zero-copy role-swap promotion.

### Unit tests
- empty-main promotion;
- full-main promotion with victim;
- READY promotion;
- QUEUED promotion + priority escalation;
- INFLIGHT promotion;
- unused READY discard;
- unused QUEUED cancellation;
- no MAIN/PREFETCH duplicate key;
- role counts preserved across 10k randomized operations.

### Exit gate
CPU fuzz + single-GPU physical slot-content validation.

---

## M5 — Decode-only boundary integration

### Work
- prefill predictor OFF;
- prefetch slots empty during prefill;
- prefill->decode state transition;
- decode begins with retained MAIN cache and empty PREFETCH slots.

### Tests
- old prefill route/cache trajectory unchanged at P=0;
- prefetch buffer empty after prefill;
- balanced prefill miss-fetch invariant.

### Exit gate
Short full prefill + decode token parity.

---

## M6 — Compact replicated metadata path

### Work
Implement one fixed-size metadata record and one collective.
Reconstruct identical demand on every rank.

Run old and new metadata paths side-by-side in validation mode.

### Tests
Per layer assert equality of:
- active experts;
- per-rank counts;
- global gate means;
- owner inputs;
- derived token->rank send/recv counts.

### Metrics
- metadata bytes;
- collective time;
- CPU pack/unpack time.

### Exit gate
100% decision-input parity and lower metadata overhead.

---

## M7 — Split controller: current critical vs next-layer prefetch

### Work
Expose:
- `plan_current(...)`;
- `plan_prefetch_next(...)`.

Current controller returns enough information to enqueue urgent H2D before the
prefetch controller runs.

### Tests
- P=0 plan parity against old controller;
- replicated plan hash equality;
- no cache drift.

### Metrics
- current-controller p50/p90;
- predictor p50/p90;
- prefetch-placement p50/p90.

### Exit gate
Current controller does not regress materially from the optimized baseline.

---

## M8 — Priority asynchronous H2D scheduler

### Work
Replace caller-thread pageable->pinned staging with workers.
Queues:
- URGENT;
- BACKGROUND.

Preserve per-slot fetch/compute events.

### Unit tests
- urgent overtakes queued background;
- in-flight copy is never overwritten;
- stage buffer reused only after DMA;
- promoted queued prefetch escalates;
- duplicate demand fetch impossible.

### Microbench
- 1..N queued experts;
- urgent-arrival while prefetch queue exists;
- H2D overlap with synthetic NCCL/GEMM.

### Exit gate
Measured urgent H2D starts earlier than old runtime and wrong prefetch cannot
increase urgent queue latency beyond defined noise tolerance.

---

## M9 — Prefetch rank placement physical integration

### Work
Enable BR/CA/LA prefetch placement into P slots.

Important:
- same predictor;
- same P;
- same balanced per-rank prefetch quota;
- only owner choice changes.

### Tests
- deterministic rank assignment;
- same candidate set across policies;
- no duplicate global prefetch;
- promotion owner equals prefetched rank.

### Exit gate
Short GPU replay parity for all three policies.

---

## M10 — Ready-first expert compute overlap

### Work
After forward receive:
- compute all ready experts first;
- only wait on a required slot when no other ready work remains.

### Validation
Compare against expert-ID-order execution using exactly the same plan.

### Metrics
- demand H2D hidden behind compute;
- stall waiting on late experts;
- count of ready experts executed before first wait.

### Exit gate
No numerical difference; H2D exposed time non-increasing.

---

## M11 — Fused token->rank 2-round A2A

### Work
Implement:
- one fused forward packet;
- destination partial aggregation;
- one return payload A2A;
- source index_add combine.

Reuse global metadata to derive split sizes.

### Tests
- reference communicator numerical parity;
- same token/expert/weight contribution;
- multiple experts for one token on same destination rank;
- zero-send peer cases;
- uneven counts;
- B128/B256;
- Env1/Env2.

### Hard gate
Trace/counters prove per MoE layer:
- exactly 1 forward payload A2A;
- exactly 1 return payload A2A;
- no count A2A;
- no hidden/weight/metadata payload A2A.

---

## M12 — Integrated overlap runtime

### Preferred ordering

```
metadata
current controller
urgent H2D submit
forward A2A async
  || predictor
  || prefetch placement
  || background H2D
forward complete
ready-first compute || remaining H2D
return A2A
combine
```

Retain diagnostic fetch-barrier mode but never use it as the optimized baseline.

### Exit gate
Full short-horizon token/cache parity and no NCCL/H2D deadlock in repeated runs.

---

## M13 — Prefetch trigger and budget physical sweep

### Matrix
- trigger T0/T1/T2;
- P near CPU knee;
- B128/B256;
- BR initially.

After selecting trigger/P:
- BR / CA / LA comparison.

### Primary selection
Minimize median TPOT subject to:
- zero correctness failures;
- bounded extra HBM;
- no urgent-H2D regression;
- acceptable wasted-prefetch bytes.

### Exit gate
Freeze one P and trigger per batch or one shared setting if results support it.

---

## M14 — Full phase-exposure study

### Work
Collect unprofiled repetitions for speed claims.
Collect separate Nsight runs for attribution.

### Report
- TPOT;
- H2D total/exposed/hidden;
- H2D hidden ratio;
- forward A2A;
- expert compute;
- return A2A;
- current controller;
- prefetch controller total/exposed;
- prefetch useful/wasted bytes;
- promotion reload cost.

### Exit gate
`RESULTS.md` explains where every material TPOT improvement comes from.

---

## M15 — Final BR/CA/LA evaluation

Use the exact same:
- predictor;
- P;
- trigger;
- C cache capacity;
- H2D scheduler;
- communicator.

Only placement objective differs.

Primary questions:
1. after H2D exposure is reduced, how much communication headroom remains?
2. how much critical-rank compute headroom remains?
3. does CA or LA dominate at B128/B256?
4. is a joint objective justified by the measured tradeoff?

Do not design the joint method before these answers exist.
