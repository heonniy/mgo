# mgo_v2

Clean research runtime for multi-GPU MoE CPU offloading.

This directory intentionally does **not** modify the two legacy MoE-Infinity trees in-place. The old repository contains a partially migrated controller-owned slot executor in C++ but still routes through the previous Python cache / dispatcher path. mgo_v2 treats the legacy C++ slot executor only as a low-level data plane and rebuilds the control plane around the final research design.

## Final research pipeline

Per global layer event:

1. collect router metadata from all ranks;
2. classify global expert Hit / Miss;
3. expert-level substitution:
   - any missed source expert with a route weight >= 0.20 is protected exactly;
   - low-importance missed experts first reuse active exact/protected anchors;
   - otherwise use the best safe resident expert with similarity >= 0.65;
   - one source expert maps to one target for all of its routes;
4. admit residual exact misses with a pluggable rank-placement policy;
5. evict with one of:
   - LRU,
   - gate-score W=128,
   - diversity-preserving W=128 / k=1 / lambda=2;
6. dispatch each token at most once per destination rank with NCCL all-to-all;
7. execute rank-local cached/fetched experts through the legacy fixed-slot executor;
8. return partial token outputs and sum them at the token-origin rank.

## Admission policies

Implemented policy interfaces include:

- balanced random baseline;
- greedy current communication;
- current + inter-layer path affinity;
- Hungarian current communication;
- Hungarian + same-layer co-activation;
- Hungarian + same-layer + inter-layer path;
- pair-swap refinement using exact deduplicated token->rank communication.

All primary policies support hard per-event balanced rank quotas.

## What is deprecated

Do not use these legacy mechanisms as the cache authority when mgo_v2 is active:

- DeviceMapManager random placement;
- ExpertPrefetcher cache replacement;
- old ExpertCache / ExpertPriorityScore;
- autonomous Archer sparse eviction;
- RPC-based distributed expert execution.

Only one controller may own residency.

## Runtime status

The policy/controller layer is self-contained and unit-testable on CPU. The distributed runtime uses torch.distributed and the legacy pybind slot executor adapter. It is intended for staged validation:

1. CPU policy tests;
2. exact EP parity;
3. controller-owned cache;
4. substitution;
5. diversity eviction;
6. admission policies;
7. physical timing.

See MIGRATION.md for the legacy issues this replaces.
