# PLAN — decode-only prefetch + cache/controller/transport refactoring

## 1. Objective

Refactor the physical MoE offloading runtime so that the dominant exposed
CPU->GPU expert-transfer latency is hidden behind useful GPU communication and
expert computation, while keeping the multi-GPU placement problem visible and
cheap enough to solve online.

Primary optimization target:

```
TPOT_decode
= metadata
+ current controller
+ exposed H2D
+ forward EP communication
+ expert compute
+ return EP communication
+ combine
```

The runtime should minimize:
- exposed H2D;
- metadata/controller overhead;
- EP communication;
- critical-rank expert compute.

Replication is removed from the active design.

## 2. Primary experiment scope

Initial matrix:
- R = 8;
- local decode batch = 128 and 256;
- global expert-cache ratio = 60%;
- Qwen3-30B-A3B exact expert routing;
- top-k = 8;
- decode horizon = 256;
- substitution OFF;
- BR / CA / LA admission;
- prefetch budget P swept separately.

Prefill:
- predictor OFF;
- prefetch OFF;
- existing main-cache path;
- mandatory misses distributed under balanced per-rank quotas.

The new method claims decode/TPOT improvement first. Prefill remains a
correctness and E2E component, not a new predictor target in this refactor.

## 3. Architectural sources to reuse

### From main: MoE-Infinity / Archer

Use as design references:
- `MoE-Infinity-EP-archer-coslot/moe_infinity/memory/expert_tracer.py`
  for trace/calibration concepts;
- `.../memory/expert_predictor.py` for prediction workflow;
- `.../memory/expert_prefetcher.py` and
  `.../distributed/expert_prefetcher.py` for asynchronous prefetch and
  prefetch-vs-demand protection semantics;
- Archer `core/prefetch/*` task scheduler / worker design for background
  pageable->pinned->H2D work;
- `MoE-Infinity-EP-coslot/moe_infinity_ep/controller/demand_collector.py`
  for replicated identical global demand;
- `.../controller/slot_cache.py` for deterministic slot-state invariants;
- `.../exec/nvlink_router.py` for derived split counts and fused payload
  forward/return A2A.

Do not copy the old trace-nearest-neighbor predictor verbatim. The refactor uses
a cheaper batch-aggregated next-layer transition predictor.

### From current mgo_v2

Reuse:
- `mgo_v2/controller.py`: deterministic replicated BR/CA logic;
- current LA implementation from the active experiment branch;
- `mgo_v2/pinned_h2d.py`: CUDA copy stream and per-slot events;
- `examples/env_offload_worker.py`: physical model harness;
- current exact route / cache trajectory validation infrastructure.

Replace or extend:
- `mgo_v2/communicator.py` final fast path;
- synchronous CPU staging in the hot thread;
- monolithic current/prefetch controller work;
- cache model so MAIN and PREFETCH roles can share one C+P physical arena.

## 4. Next-layer predictor

### 4.1 Calibration

For token t and layer l, record routed top-k set:

```
S[t,l]
```

Build per-layer transition counts:

```
C_l[i,j] = # tokens where i in S[t,l] and j in S[t,l+1]
```

and normalized transition matrix:

```
T_l[i,j] = C_l[i,j] / sum_j C_l[i,j]
```

Calibration and evaluation request sets must be disjoint.

### 4.2 Batch runtime prediction

Each rank has current-layer expert histogram:

```
h_l[r,i] = local route count for expert i
```

All ranks receive the same global metadata, so every rank reconstructs
`h_l[R,E]`.

Predicted next-layer per-rank demand:

```
Dhat[l+1,r,j] = sum_i h_l[r,i] * T_l[i,j] / top_k
```

Also compute global predicted demand:

```
Ghat[j] = sum_r Dhat[r,j]
```

Candidate filtering happens against the **post-current-controller main-cache
shadow**, not the pre-controller snapshot.

Exclude:
- experts already MAIN-resident globally;
- an expert already reserved in a prefetch slot;
- invalid target layers;
- layer l+1 beyond the final MoE layer.

### 4.3 Prediction evaluation

Before physical integration, measure:
- Precision@P;
- Recall@P;
- actual-miss Recall@P;
- demand-weighted recall;
- per-rank demand MAE;
- owner regret for CA/LA;
- oracle predictor upper bound.

The primary predictor metric is actual-miss recall, not generic activation
accuracy.

## 5. Prefetch budget

The runtime parameter is **P slots per rank**.

Physical expert arena per rank:

```
Nslots = C(cache-ratio-derived) + P
```

Initial sweep:
- P = 0, 1, 2, 4, 8.

Do not pick P by prediction hit rate alone. Select the smallest P near the TPOT
knee after physical measurement.

Also report extra HBM:

```
extra_HBM_per_rank = P * expert_bytes
```

## 6. Prefetch rank placement

A predicted expert has one global prefetch owner.

All policies receive the same candidate set and the same balanced prefetch
quota:

```
max(prefetch_count_by_rank) - min(prefetch_count_by_rank) <= 1
```

### BR-prefetch

Deterministic balanced-random assignment.

### CA-prefetch

Use predicted per-rank demand and assign candidates to balanced rank quota
slots to maximize predicted local service:

```
maximize sum_e Dhat[owner(e), e]
```

### LA-prefetch

Estimate predicted next-layer load from:
- next-layer experts already MAIN-resident;
- candidate prefetch placements.

Assign under the same balanced prefetch quota to minimize:

```
max_r predicted_expert_rows[r]
```

Tie order:
1. lower predicted max load;
2. lower destination predicted load;
3. larger predicted local demand;
4. lower rank id.

No future actual routing may be read.

## 7. MAIN + PREFETCH unified physical arena

### 7.1 Slot roles

Each rank allocates C+P physical expert-sized slots.

Each physical slot has:
- role: MAIN or PREFETCH;
- key: optional (layer, expert);
- transfer state;
- CUDA fetch event;
- cache metadata if MAIN;
- target-layer / predictor score if PREFETCH.

Exactly C roles are MAIN and exactly P roles are PREFETCH at all times.

### 7.2 Why role swap

Copying a correctly prefetched 9-MiB expert from a prefetch buffer into a
separate main-cache slot would reintroduce avoidable device traffic.

Instead promotion changes metadata ownership, not tensor contents.

### 7.3 Promotion

At target layer routing:

1. classify normal MAIN hits;
2. match demand against PREFETCH reservations;
3. for every demanded prefetched expert:
   - force execution owner to the prefetched rank;
   - select a legal MAIN victim (or empty MAIN-role slot);
   - swap physical slot roles;
   - update identical logical main-cache shadow on all ranks;
   - reuse the existing fetch event if transfer is incomplete.

Result:

```
prefetch physical slot -> MAIN
victim/empty main physical slot -> PREFETCH_EMPTY
```

No expert-weight device copy.

### 7.4 Promotion and cache policy

A promoted expert receives normal MAIN cache metadata:
- admitted_at = promotion tick;
- last_used = current tick;
- gate/LRU state initialized as a real current demand.

The evicted victim follows normal main-cache accounting. Future reloads caused
by promotion must be measured.

### 7.5 Wrong predictions

Unused prediction:
- QUEUED: cancel before staging if possible;
- INFLIGHT: finish safely, then invalidate;
- READY: invalidate immediately.

Wrong prefetch never directly evicts MAIN because PREFETCH slots are dedicated.

## 8. Compact metadata exchange

The current `gather_global_routes` transports more data than the fast
controller requires.

For fixed-size decode batches, pack one per-rank metadata buffer containing:

1. selected expert IDs per local token, preferably uint8/uint16;
2. optional per-token validity count if required;
3. sum of full router probabilities per expert for exact gate-history update;
4. small version/step/layer header.

All ranks all-gather the same fixed-size record once.

From it every rank reconstructs:
- per-rank expert counts;
- global active expert set;
- token-level expert combinations needed to derive coalesced A2A counts;
- global gate-probability mean;
- predictor batch histogram.

No controller plan broadcast in production.

Debug mode:
- compute deterministic plan hash;
- tiny hash all-gather;
- fail on mismatch.

## 9. Controller split

Do not keep one monolithic controller call.

### 9.1 Current-layer critical controller

Runs immediately after metadata arrives.

Order:
1. apply target-layer prefetch promotions;
2. main hit/miss classification;
3. BR/CA/LA assignment for residual demand misses;
4. victim/slot decisions;
5. mutate replicated main-cache shadow;
6. return urgent H2D operations and owner map.

As soon as this returns, submit current demand H2D.

This controller is fully on the critical path and must be minimized.

### 9.2 Next-layer prefetch controller

Runs after current urgent H2D submission and preferably after asynchronous
forward A2A launch.

Order:
1. matrix prediction;
2. candidate filtering on post-current-plan MAIN state;
3. top candidate selection subject to P;
4. BR/CA/LA prefetch owner decision;
5. reserve PREFETCH slots;
6. enqueue background H2D.

Its CPU time should overlap forward A2A whenever possible.

## 10. H2D scheduler

### 10.1 Current problem

The current Python pinned path performs pageable->pinned staging synchronously
on the caller thread. That delays communication launch and prevents true early
H2D.

### 10.2 Refactored scheduler

Use worker-driven staging, inspired by Archer's prefetch task scheduler.

Two priority queues:

```
URGENT:
- current residual demand misses
- demanded prefetched expert whose copy has not started

BACKGROUND:
- unused next-layer predicted prefetch
```

Worker pipeline:
1. reserve pinned stage;
2. pageable/file-backed expert -> pinned copy on worker CPU;
3. submit pinned -> GPU on H2D stream;
4. record slot event;
5. recycle stage only after DMA completion.

### 10.3 Priority contract

If urgent work exists, no not-yet-started background prefetch may start first.

A currently INFLIGHT background prefetch is not forcibly canceled.

### 10.4 Early start point

Urgent current H2D submission must occur immediately after the current
controller emits its plan, before forward A2A waits.

## 11. H2D / communication / compute overlap

Optimized ordering:

```
current controller
  -> urgent H2D queued
  -> fused forward A2A async
       || current H2D
       || next-layer predictor/controller
       || next-layer background H2D
  -> forward receive ready
  -> compute every expert whose slot is ready
       || remaining urgent H2D
       || background H2D
  -> late experts wait only on their own slot events
  -> return A2A
  -> combine
```

No global H2D barrier.

### Ready-first expert scheduler

Never iterate expert IDs blindly if the first expert is waiting on H2D while
another required expert is already ready.

Execution classes:
1. MAIN resident ready;
2. promoted prefetch ready;
3. newly fetched miss ready;
4. pending required experts.

Run ready classes first. Wait only when no executable expert remains.

## 12. Two-round expert communication

The fast expert-execution path has exactly two payload A2As.

### 12.1 Forward fused token->rank packet

For each local token and destination rank, create at most one packet:

```
hidden | expert_ids[K] | routing_weights[K]
```

Only experts owned by that destination rank occupy the packet.

This preserves token->rank coalescing: if two experts for one token live on the
same rank, hidden is transferred once.

All send/recv split counts are derived from globally shared routing metadata.
No count-exchange collective is required in the fast path.

### 12.2 Destination computation

For each received packet:
- execute all listed experts;
- apply routing weights;
- sum destination-rank partial contribution for that token.

### 12.3 Return A2A

Return exactly one hidden-sized partial per received forward packet, ordered
so the source already knows the token index.

No return expert-id/weight/index metadata collective.

Source performs `index_add_` into the local output.

Thus payload communication is:

```
1 x forward A2A
1 x return A2A
```

Metadata all-gather remains a separate small collective.

## 13. Prefetch trigger position sweep

Measure at least:

### T0 — before forward A2A

Current controller + prefetch predictor/placement both run before communication.
Earliest prefetch request, but delays urgent H2D / dispatch.

### T1 — preferred

```
current controller
-> urgent H2D
-> launch forward A2A async
-> predictor + prefetch controller
-> background prefetch H2D
```

This prioritizes current demand and tries to hide predictor CPU time under A2A.

### T2 — after forward A2A

Simplest but shortest prefetch window.

Select trigger by actual TPOT / H2D-exposure measurements, not intuition.

A possible later micro-optimization is splitting prediction into:
- early pure matrix prediction;
- post-current-plan candidate filtering/placement.

Do not implement this until T0/T1/T2 results justify it.

## 14. Prefill behavior

Prefill does not use the new predictor or prefetch buffers.

The current code already performs per-event balanced mandatory miss admission:
the number of miss fetches assigned to ranks differs by at most one.

BR/CA/LA may choose different experts for each quota slot, but they keep the
same balanced fetch-count constraint.

At the prefill->decode boundary:
- clear all PREFETCH reservations;
- all P prefetch slots must be PREFETCH_EMPTY;
- retain normal MAIN cache state produced by prefill.

## 15. Controller / metadata optimization targets

Measure before optimizing further.

Target properties:
- one small metadata all-gather;
- no plan broadcast;
- no per-rank RPC;
- deterministic local decision;
- current controller much smaller than saved exposed H2D;
- prefetch controller mostly hidden behind forward A2A.

If Hungarian CA is too expensive at B128/B256, preserve the exact CA baseline
but add a decision-equivalent or bounded greedy fast path only after profiling.

Do not silently replace CA semantics to gain controller speed.

## 16. Three-arm final runtime selection

The final implementation is **not pre-decided**. Build and physically compare
three runtime variants that differ only in prefetch / H2D exposure policy.

All other implementation details MUST be identical across the three arms:
- same compact one-shot metadata exchange;
- same replicated deterministic controller;
- same BR and LA placement implementation;
- same cache ratio C;
- same expert kernel and compilation mode;
- same fused token->rank packet format;
- exactly one forward payload A2A + one return payload A2A;
- same NCCL settings / communicator optimizations;
- same CPU affinity / NUMA binding;
- same frozen requests, routes, weights and teacher tokens;
- same prefill behavior;
- same timing boundaries and repeat rule.

This is required so the measured LA gain changes only because H2D exposure
changes, not because one arm has a better communicator or controller.

### V1 — OPT-NOPF-BARRIER

Purpose: optimized version of the current physical LA experiment.

Decode order:

```
metadata
-> current controller
-> fused forward A2A
-> residual demand H2D
-> global fetch barrier
-> expert compute
-> return A2A
-> combine
```

Properties:
- prefetch OFF;
- P=0;
- global fetch barrier ON;
- current-demand H2D remains fully exposed before compute;
- all metadata/NCCL/controller refactoring is ON.

This is the clean reference for the already observed physical LA mechanism.
The historical `b18d9d7` result is not the final V1 number because it did not
yet include all common refactoring optimizations.

### V2 — OPT-PF-BARRIER

Purpose: isolate the effect of next-layer prefetch while preserving the fetch
barrier.

Decode behavior:
- EdgeMoE-style batch next-layer predictor ON;
- C+P MAIN/PREFETCH arena ON;
- useful prefetch promotion ON;
- selected P/trigger fixed identically for BR and LA;
- residual current misses still pass through the global fetch barrier before
  expert compute.

Conceptually:

```
previous/current useful work
   || next-layer prefetch H2D
next layer:
metadata
-> current controller + promotion
-> fused forward A2A
-> residual demand H2D
-> global fetch barrier
-> expert compute
-> return A2A
-> combine
```

This arm answers whether reducing the amount of demand H2D alone makes LA's
compute-balancing gain more visible.

### V3 — OPT-PF-OVERLAP

Purpose: full optimized runtime.

Properties:
- same predictor, P, promotion and prefetch placement semantics as V2;
- global fetch barrier OFF;
- residual demand H2D submitted as early as possible after current-controller
  hit/miss/owner decision;
- demand H2D overlaps forward communication and ready expert compute;
- background prefetch remains lower priority than current demand;
- expert execution is ready-first with only per-slot waits.

Decode order:

```
metadata
-> current controller
-> urgent demand H2D submit --------------------------+
-> fused forward A2A async                            |
      || predictor / prefetch controller              |
      || next-layer background H2D                    |
forward receive ready                                 |
-> ready-first expert compute <-----------------------+
      || remaining demand H2D
-> return A2A
-> combine
```

This arm answers whether pushing exposed H2D below the communication / expert
compute critical path increases LA's realized TPOT benefit.

### Selection metric

For each variant v and local batch B:

```
G_LA(v,B) = (TPOT_BR(v,B) - TPOT_LA(v,B)) / TPOT_BR(v,B)
```

Only stable BR and LA measurements are eligible.

Primary final-runtime score:

```
G_robust(v) = min(G_LA(v,B128), G_LA(v,B256))
```

Choose the eligible variant with the largest `G_robust`.

Tie-breaks, in order:
1. larger mean LA gain across B128/B256;
2. lower absolute LA TPOT averaged across B128/B256;
3. lower exposed H2D;
4. lower extra HBM / wasted prefetch bytes.

The robust minimum prevents choosing a runtime that only makes LA look strong
on one batch size.

### Anti-gaming constraint

A variant may not win merely because BR became artificially slower.

Therefore report for every arm:
- absolute BR TPOT;
- absolute LA TPOT;
- LA gain;
- exposed H2D;
- communication time;
- expert-compute time;
- controller time.

If a candidate has larger relative LA gain but materially worse LA TPOT than a
strictly dominated arm, flag it as `GAIN_ONLY_DOMINATED` and do not adopt it.

A simple dominance rule is sufficient:
variant A dominates B if A has
- no worse stable LA gain (within measurement tolerance), and
- lower absolute LA TPOT.

### Prefetch tuning fairness

P and trigger timing are tuned **before** the final BR-vs-LA comparison using a
policy-neutral protocol:
- first use BR-only physical sweeps to select a small P/trigger knee;
- freeze that P/trigger;
- then run BR and LA using exactly the same frozen prefetch configuration.

Do not tune P separately for LA.

### Final adoption

After the three-arm physical comparison:
- write `FINAL_RUNTIME_SELECTION.json`;
- record all stable repeats and phase exposure;
- set the winning variant as the default optimized runtime;
- keep the other two as explicit ablation/diagnostic modes;
- do not delete fetch-barrier mode because V1/V2 remain useful mechanism
  controls.

## 17. Final success conditions

Correctness:
- exact token/output parity;
- no cache-shadow drift;
- no duplicated demand H2D;
- C MAIN + P PREFETCH role counts preserved;
- P=0 trajectory parity with baseline;
- exactly two payload A2As/layer.

Performance:
- substantial reduction in exposed H2D;
- current controller overhead does not erase H2D savings;
- predictor controller largely hidden;
- B128/B256 TPOT improves over the same optimized no-prefetch runtime;
- BR/CA/LA comparisons use the same cache ratio, P, predictor and H2D scheduler.

The final paper-quality comparison must be against an optimized baseline with
the same 2-round communicator and early demand-H2D scheduler, not a deliberately
slow baseline.
