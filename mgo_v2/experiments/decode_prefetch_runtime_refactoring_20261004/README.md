# Decode-only MoE offloading runtime refactoring

This packet is the implementation plan for the next `mgo_v2` runtime.

## Scope

Primary target:
- Qwen3-30B-A3B style MoE;
- R8 EP;
- constrained HBM expert cache;
- CPU expert offloading;
- decode only for the new prefetch pipeline;
- local batch B128/B256 first;
- substitution OFF;
- replication REMOVED from the active design.

The new runtime has four goals:

1. hide as much **H2D exposure** as possible with next-layer prefetch and
   early demand fetch;
2. keep controller/metadata overhead small and deterministic across ranks;
3. reduce expert execution transport to **one fused forward A2A + one return
   A2A per MoE layer**;
4. optimize the remaining communication / expert-compute bottleneck with
   BR, CA, and LA placement.

Prefill is deliberately kept simple and unchanged in the first implementation:
no predictor/prefetch. Existing mandatory misses retain balanced per-rank fetch
counts; BR/CA/LA may differ in which miss is assigned to which rank.

## Prefetch memory model

Each rank allocates one physical expert-slot arena with

```
C main-cache slots + P predictor-prefetch slots
```

where `C` is derived from the original cache ratio and `P` is swept.

The prefetch buffer is **not a separate disposable copy cache**. A correctly
predicted expert can be promoted into the persistent main cache without a
9-MiB device copy by atomically swapping slot roles:

```
PREFETCH_READY(E7) slot  --promote--> MAIN(E7)
MAIN(victim/empty) slot  --swap-----> PREFETCH_EMPTY
```

Thus:
- main-cache capacity remains exactly `C`;
- prefetch capacity remains exactly `P`;
- promotion adds no D2D copy;
- an unused prediction is discarded from the prefetch role;
- a pending prediction that becomes demanded is promoted/escalated and the
  existing H2D is reused rather than duplicated.

## Intended decode pipeline

```
router
  -> compact global metadata
  -> replicated deterministic current-layer controller
  -> submit current demand H2D immediately
  -> launch fused forward A2A
       || next-layer batch predictor + BR/CA/LA prefetch placement
       || background next-layer H2D
  -> ready-first expert compute
       || remaining demand H2D
       || next-layer prefetch H2D
  -> fused return A2A
  -> combine
```

There is no global H2D barrier in the optimized runtime. The old fetch-barrier
mode is retained only as a diagnostic reference.

See:
- `PLAN.md`: architecture and algorithms;
- `MILESTONES.md`: implementation order and per-stage gates;
- `INVARIANTS.md`: correctness contracts;
- `MEASUREMENT.md`: H2D/comm/compute exposure attribution;
- `matrix.json`: frozen initial experiment matrix;
- `AGENT_TASK.md`: execution instructions.
