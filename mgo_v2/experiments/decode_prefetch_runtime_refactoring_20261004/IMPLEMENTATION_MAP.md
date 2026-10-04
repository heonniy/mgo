# Implementation map

This file maps planned functionality to concrete modules so refactoring does
not produce another parallel code tree without ownership.

| Concern | Existing reference | Planned owner |
|---|---|---|
| MAIN cache state | `mgo_v2/cache.py` | refactor into unified C+P slot arena |
| Prefetch state | Archer/MoE-Infinity prefetcher | new `mgo_v2/prefetch.py` or equivalent single owner |
| Transition predictor | main expert tracer/predictor | new CPU-only calibration + lightweight runtime predictor |
| Current BR/CA/LA | controller/admission code | keep in authoritative controller |
| Prefetch BR/CA/LA | none final | same placement interface, predicted-demand context |
| Metadata | `runtime.gather_global_routes` / main DemandCollector | new compact fixed-record collector |
| H2D | `pinned_h2d.py` | extend to worker-driven priority scheduler |
| Forward/return comm | `communicator.py`, main `nvlink_router.py` | one new fused 2-round fast path |
| Physical harness | `env_offload_worker.py` | keep as integration/timing harness |
| Phase analysis | existing Nsight scripts | add interval-union exposure analyzer |

## Avoid duplicate authorities

There must be exactly one authoritative object for each state:

- MAIN/PREFETCH logical slot state: controller cache arena;
- physical expert bytes: GPU arena;
- transfer readiness: rank-local H2D scheduler events;
- predictor table: immutable calibration artifact;
- current layer plan: deterministic controller result;
- communication packet layout: fast communicator.

Do not maintain a second independent cache dictionary in the worker.

## Recommended new interfaces

### Cache arena

```python
arena.main_owner(key) -> rank | None
arena.prefetch_owner(key) -> rank | None
arena.reserve_prefetch(rank, key, target_layer, score)
arena.promote_prefetch(key, victim_policy) -> Promotion
arena.discard_prefetch(key)
arena.snapshot_hash() -> bytes
```

### Predictor

```python
predictor.predict_next(layer, per_rank_counts) -> predicted_demand[R,E]
```

No CUDA tensor allocation in predictor hot path after initialization.

### Prefetch planner

```python
planner.plan(predicted_demand, cache_snapshot, P) -> PrefetchPlan
```

Same planner interface for BR/CA/LA.

### H2D scheduler

```python
scheduler.enqueue_demand(slot, key, tensors)
scheduler.enqueue_prefetch(slot, key, tensors)
scheduler.promote(slot)
scheduler.cancel_if_queued(slot)
scheduler.ready(slot) -> bool
scheduler.wait(slot)
```

### Fast communicator

```python
packet = comm.forward_async(hidden, selected, weights, owner_map, global_metadata)
received = packet.wait()
returned = comm.return_async(partials, packet)
output = returned.wait_and_combine()
```

The communicator must expose operation counters so the 2-A2A invariant is
machine-verifiable.
