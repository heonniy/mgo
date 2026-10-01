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

## Must be validated on the current server before paper timing

1. build the legacy extension and run exact EP parity;
2. validate per-rank slot capacity and cache-state parity;
3. validate expert-level Hit/SubHit/Miss against the completed simulator;
4. validate R4 then R8 NCCL dispatch/combine;
5. profile controller overhead;
6. remove the resident-slot -> MoEMLP parameter D2D copy on every hit;
7. only then collect real H2D / NVSwitch / end-to-end latency tables.

## Why the D2D copy is not patched blindly here

The current C++ MoEMLP owns reusable parameter tensors and copies slot data
into them before every GEMM. Converting those tensors into direct views of
slot memory changes lifetime/aliasing assumptions in the CUDA extension.
That change needs compile + CUDA correctness testing on the target H100
server. mgo_v2 isolates the issue and records it as the final low-level
performance blocker rather than silently changing CUDA ownership semantics
without a runnable server.
