# Controller overhead repair — 2026-10-02

**Status: complete, PASS.** Read [RESULTS.md](RESULTS.md) and
[validation.json](validation.json). All nine primary generations, 9,360 CPU
differential events, two short GPU profiles and 39 CPU tests passed. No OOM or
memory-guard stop occurred. Do not automatically expand or repeat the packet.

This packet follows the completed rank-demand oracle packet at `c074593`.
Its scope is the owner's minimum sufficient confirmation: R4/B8 only, one
65-forward generation per P0/P1/O0 policy and C0/C1/C2 implementation (nine
generations total). No controller expansion to B4/B16 is authorized by this
packet. See [PROTOCOL.md](PROTOCOL.md) and the original packet's
[scope amendment](../rank_demand_oracle_20261002/scope_amendment.json).

- **C0:** unchanged original controller, with fresh control measurements.
- **C1:** incremental Coverage/cache indexes and vectorized victim ranking;
  identical candidates, decisions, tie-breaking and history rounding.
- **C2:** C1 planning on rank 0, followed by a compact decision broadcast and
  exact follower application. All ranks still supply routing metadata.

Correctness uses complete captured-route C0/C1/follower CPU replay, offline
cache/history/index checks, original event/cache/token parity on every physical
rank, and device-work accounting. CPU component instrumentation is separate
from primary timing. Two short P1 profiles (prefill plus eight decode forwards)
follow timing and reuse the matching prefix of the original C0 profile.

Timing is descriptive: one sample per cell, ordered C0 then C1 then C2 on a
shared host. This design establishes neither repeatability nor small speedups.
The optimized paths remain opt-in; no placement or substitution policy changes.

Only physical GPUs 0,1,4,5 are used, with one model job at a time and host/GPU
memory guards. Large raw captures remain outside Git at
`/home/hwlee/mgo-results/controller_overhead_20261002`.

The frozen source and protocol receipt is [measurement_manifest.json](measurement_manifest.json).
Final outputs are `RESULTS.md`, `validation.json`, `e2e_comparison.csv`, CPU
component summaries, GPU profile summaries and provenance/hash receipts.
