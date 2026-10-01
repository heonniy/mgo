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

The final R4/R8 matrix completes 60 conditions × two repeats, including all
batch/cache cells and A/B/C ablations. Every rank receipt is audited; native
quality controls use the same local batch as the corresponding measurement.
Complete R4/R8 Nsight diagnostics audit 120 rank/cell ranges, with exact
physical expert-transfer accounting and unchanged policies/generated outputs.
The report retains two-repeat ranges, answer gains/losses, controller overhead
and the fixed 16-step/zero-shot-screen limitations. Communication reduction
alone does not imply a throughput improvement in these measurements.

The runtime fingerprint in the checked-in provenance identifies the measured
sources and compiled extension at `a0be82e`. Later report/analysis changes do
not alter that runtime. Standard GSM8K test accuracy, longer generations and
statistical confidence beyond two repeats are not established here.

## Direct-slot execution

The tensor index now points at the actual CUDA slot. Native execution uses
those views until the controller's pinned event finishes; CUDA events protect
reuse and return-stream access. An explicit drained reset restores host views
before freeing/resizing the pool. Nsight confirms zero expert-sized D2D copies
in direct-view mode and H2D bytes equal logical misses. Copy mode remains an
explicit profiling baseline. Final model timing and quality qualifications
belong to the source-backed validation report, not to a microbenchmark claim.
