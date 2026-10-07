# MAIN eviction-policy study — 2026-10-07

## Question

Why is the current MAIN distinct-expert hit rate low, and can a different
victim policy improve it without changing admission/communication policy?

This packet isolates **MAIN eviction**. It does not change the production
runtime by default and it does not claim TPOT improvement until a winning
policy is physically validated.

## Fixed scope

- Qwen3-30B-A3B-Instruct-2507, BF16
- R4 physical GPUs **0,1,4,5 only**
- C30
- input 256, output 64
- local batch **B8, B16, B64** (global 32/64/256)
- target request order inherited from the frozen main-table ShareGPT workload;
  B8 takes the first 8 requests from each B16 rank shard
- route capture source: current LA_CA_NEAR selected H0/full-pinned runtime
- dynamic expert state starts cold before the measured generation
- one route-capture model run per batch; all eviction policies consume the exact
  same frozen global demand/gate stream

The simulator uses the production MAIN capacities [459,459,459,458]. The two
physical prefetch slots/rank are deliberately **not simulated**. This is a P0
MAIN-only counterfactual so that prefetch promotion/eviction cannot hide the
effect of the victim policy. Admission is fixed to LA_CA_NEAR (near_bps=200).

## Policies

1. **gate_w128** — current victim rule: minimum per-layer Gate-W128 score,
   tie by older last-use then expert key.
2. **lfu_reset** — LFU count is one per distinct expert event and is reset when
   the expert leaves MAIN.
3. **lfu_cumulative** — the same LFU count persists across eviction/reload.
4. **lru_reset**
5. **lru_cumulative**

For LRU, a reload is itself an access at the current tick. Therefore standard
LRU has no accumulated counter to preserve: reset and cumulative labels are
mathematically identical under this runtime. The replay keeps both requested
labels and asserts exact equality instead of inventing a non-standard LRU.

LFU is deliberately **distinct-expert frequency**, not token-weighted
frequency, because the primary metric is MAIN distinct-expert hit.

## Metrics

For decode only (prefill warms the initially empty MAIN cache):

- MAIN distinct-expert hit rate, first decode step and all decode
- token-weighted MAIN hit rate as a secondary metric
- demand misses/fetches
- first fetches vs reload fetches
- evictions
- per-step hit trajectory
- final cache-state digest

A MAIN hit means the layer/expert is already in MAIN immediately before the
current event's admissions. There is no PREFETCH-hit category in this isolated
replay.

## Execution

Run:

`python mgo_v2/scripts/run_main_eviction_policy_study.py`

The script creates one B8 manifest by rank-preserving slicing, launches only
three model route captures (B8/B16/B64), then runs all five eviction policies
CPU-only over each trace. It writes B8/B16/B64 simulation JSONs here.

Do **not** run GPUs 2,3,6,7. Do not replace the production eviction policy from
simulation alone. If a policy materially improves MAIN hit, validate only that
winner physically in a later explicit owner-approved experiment.
