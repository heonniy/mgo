# Reference Oracle — encoded SPEC

This oracle (`reference.py`) is the **new ground truth** for controller-side
cache planning. It is derived from the design spec, **not** from the production
`moe_infinity_ep.controller` code and **not** from the suspect golden JSONL in
`실험/moe_cache_golden`. The old golden files are treated as untrusted.

## Decisions encoded

| Decision | Rule |
|---|---|
| ExpertKey | `(layer_id, expert_id)` |
| naive owner | `owner(expert) = expert_id % ep_size` |
| balanced owner | iterate `sorted(miss_keys)`; assign each to `argmin (load, rank)`, `load` = #assigned this layer, `+1` per assignment |
| LRU victim | `argmin (last_used, inserted_at, slot)` |
| LFU victim | `argmin (freq, last_used, inserted_at, slot)` |
| hit touch | a resident hit refreshes `last_used` and adds demand to `freq` **before** Phase-2 victim selection |
| install | freshly installed expert gets newest `last_used` (counter bumped on apply) and `freq = its demand` |
| plan order | per rank, misses processed by `(-demand, layer, expert)`; lowest-index empty slot preferred before any eviction |
| locate | expert resident in multiple ranks resolves to the **lowest** rank index |

## Why these (and how to challenge them)

The oracle catches **implementation drift** (production deviating from the spec
above). It cannot by itself catch a wrong *spec*. Two extra guards do:

1. `test_reference_selftest.py` — micro-traces whose answers are **hand-computed
   in the comments**, anchoring the oracle to human reasoning, not to either
   codebase.
2. Invariant tests (`B*`, `H*` in the plan) — balance ≤ 1, accounting sums,
   no duplicate fetch, slot-drift = 0 — derived from first principles.

If a differential test fails, triage is: *is the oracle wrong, the production
code wrong, or the spec ambiguous?* — resolve against the design doc, never by
blindly matching either side.

## EAMC priority eviction (added 2026-06-03)

The controller-side EAMC victim selection is now implemented and modelled:

| Decision | Rule |
|---|---|
| EAMC victim | `argmin (priority[l][e], last_used, inserted_at, slot)` over occupied |
| protect | experts demanded THIS layer (`key ∈ demand`) are excluded (원본 `protected_ondemand`); if all are protected, force over all |
| priority None | fall back to LRU (aggregator-off / stride-skip safety) |

`priority` is a `[num_layers, num_experts]` matrix (higher = more important).
Production `PriorityEviction` (evict_policy.py, name `eamc`) consumes it; the
controller injects it per layer via `set_layer_priority` / `plan_misses(priority=)`.
The oracle's `priority_victim` is the independent cross-check; see
`test_eamc_eviction.py` (hand-traces + 900 random differential traces).

**Integration boundary (model-dependent, NOT covered here):** the priority
MATRIX source — constructing `ExpertPredictor` from a saved EAMC trace, running
`PriorityAggregator.aggregate` per layer with seq-ids, and calling
`set_layer_priority` before `plan_misses` — needs a loaded model + saved trace,
so it is verified at the benchmark level, not in this CPU suite. The oracle only
asserts the VICTIM-SELECTION semantics given a priority matrix.

## What is intentionally NOT modelled

* The EAMC priority MATRIX computation (`topo × decoder × frequency` via the
  upstream predictor/tracer) — needs the predictor + trace; only victim
  selection given a matrix is modelled.
* DemandAware owner/evict — exist in production but secondary; not the oracle's
  focus.
* Byte-budget eviction — slots are the unit; byte limits live in the C++ archer.
