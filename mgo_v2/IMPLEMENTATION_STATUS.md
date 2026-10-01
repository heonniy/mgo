# Implementation status

## Implemented in mgo_v2

- explicit global expert-slot capacities;
- single-copy rank ownership;
- expert-level cache-aware substitution (.20 gate protection / .65 similarity);
- route merging after substitution;
- LRU eviction;
- Gate W128 eviction;
- Diversity Coverage W128 / k=1 / lambda=2 eviction;
- balanced-random admission;
- greedy communication admission;
- greedy communication + path admission;
- Hungarian current-communication admission;
- Hungarian same-layer affinity admission;
- Hungarian same-layer + inter-layer path admission;
- quota-preserving exact pair-swap refinement;
- one-token-per-destination-rank NCCL dispatch representation;
- NCCL return/combine path;
- rank-local adapter for the controller-owned C++ slot executor;
- Qwen3 MoE forward patch hook;
- current-server GPU/NUMA auto-discovery and CPU affinity pinning;
- pure-Python regression tests for policy semantics.

## Deliberately disabled

- old DeviceMapManager;
- old ExpertCache / ExpertPriorityScore as cache authority;
- old ExpertPrefetcher as cache authority;
- Archer autonomous sparse eviction;
- RPC expert dispatch.

## Current-server validation

The H100 server now builds and runs the isolated `_store` data plane. Native
full-model R1/R4/R8 parity at 10% cache, physical tensor-address/fetch audits,
policy replay, empty-rank/custom-stream fixtures and direct slot views pass.
See [SERVER_VALIDATION_RESULTS.md](SERVER_VALIDATION_RESULTS.md) for exact scope,
raw receipts and measurement status.

The following gates govern paper timing:

1. build the legacy extension and run exact EP parity;
2. validate per-rank slot capacity and cache-state parity;
3. validate expert-level Hit/SubHit/Miss against the completed simulator;
4. validate R4 then R8 NCCL dispatch/combine;
5. profile controller overhead;
6. remove the resident-slot -> MoEMLP parameter D2D copy on every hit;
7. only then collect real H2D / NVSwitch / end-to-end latency tables.

## Direct-slot execution

The tensor index now points at the actual CUDA slot. Native execution uses
those views until the controller's pinned event finishes; CUDA events protect
reuse and return-stream access. An explicit drained reset restores host views
before freeing/resizing the pool. Nsight confirms zero expert-sized D2D copies
in direct-view mode and H2D bytes equal logical misses. Copy mode remains an
explicit profiling baseline. Final model timing and quality qualifications
belong to the source-backed validation report, not to a microbenchmark claim.
