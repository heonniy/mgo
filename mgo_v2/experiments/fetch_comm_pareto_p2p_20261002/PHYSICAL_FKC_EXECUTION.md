# Frozen six-cell execution receipt conventions

Plan: `145b3e8`, `PHYSICAL_FKC_PILOT.md`. No policy retuning or production
controller change. F/K/C schedules use the unchanged CPU implementation from
`dc7b099`. The three compressed schedules remain outside Git with exact paths,
SHA256 hashes, sizes and CPU-total parity in `physical_fkc_schedule_metadata.json`.
They are read-only; each worker verifies its schedule hash before loading a model.
All 1,296 scheduled events passed an independent action-applier/state check.
Two additional CPU tests verify valid replay and reject corrupted victims,
fetch classes, destinations, slot plans and state hashes.

Each schedule includes raw routes/origins, every admission (including its fetch
class and eviction), physical slots/copy sets, route destinations, rank-local
hit/miss operations, dispatch/combine matrices and complete post-state hash.
Every rank applies the same frozen actions. The existing dispatcher can use an
origin-specific owner map: all tokens in one process have the same origin rank,
so the plan's local-first/primary rule maps each expert to one destination for
that process. No online greedy search or production owner-map change is needed.

A worker checks raw route and origin equality before applying an event, actual
received and returned token/expert identities, physical cache slots after every
execution, physical fetch increments and actual send/receive count matrices.
The cross-rank cell validator checks matrix transposes, state hashes, all fetch
classes, activation totals, and generated-token equality to the source and all
completed cells. Any failure stops the packet; partial receipts are retained.

Primary timing is one generation per cell, no model warmup or retries, with
one prefill plus eight decode forwards. Report maximum cumulative rank decode
duration and divide by eight for primary ms/step; also retain per-rank/per-step
raw times. E2E is maximum rank wall time. All primary times include obligatory
per-event validation and frozen action application. Application CPU timing
includes the logical action/state checks and route-map construction, excluding
routing collectives and physical execution. No policy search is timed.

Existing lightweight collective CUDA-event intervals and native dispatcher
host counters are enabled identically across cells. Native phase counters are
fetch wait, weight binding, compute+sync, and combine **host** microseconds;
they are not isolated H2D durations or GPU kernel times. MoE current-stream
CUDA intervals include launch/CPU gaps and waits, not kernel-only duration.
Isolated H2D GPU and expert-kernel timings are omitted; no profiler is added.
Collective timings include self/metadata work even when peer activation bytes
are zero. Interval sums may overlap and must not be added as a critical path.

One 32-KiB-per-peer transport preflight per mode reuses the validated worker.
Only these preflights enable NCCL INFO. Require P2P under T0 and
SHM/direct/direct under R3, with no NET fallback. Timed workers assert the
exact prescribed environment and have INFO disabled.

Use only GPUs 0,1,4,5. Launch requires >=512 GiB host available and <1 GiB
used on each target GPU. Poll every five seconds and stop if host available
falls below 128 GiB, worker-tree RSS exceeds 320 GiB, or a target GPU has
<8 GiB free. Each cell is bounded at 900 seconds including load; no retry.
The normal direct CPU-to-slot path is required; staging-copy count must stay
zero, and slot views avoid full-expert parameter copies.

Run with `/home/hwlee/sub-moe/phase01/.venv/bin/python
mgo_v2/scripts/run_physical_fkc_pilot.py` from the repository root. Cell
order is T0-F, R3-C, T0-K, R3-K, T0-C, R3-F; existing output directories
are never overwritten. CPU schedule generation is not repeated for timing.
