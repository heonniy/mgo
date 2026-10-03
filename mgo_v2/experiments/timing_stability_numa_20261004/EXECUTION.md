# Execution freeze — 2026-10-04

The old policy matrix is paused and must not resume automatically. Ten completed
MEASURE receipts are retained but timing-invalid for policy comparison. During
transition, the STOP marker interrupted active P/CA/Env2 repeat 2 rather than
waiting for completion as requested; that partial run is retained as FAIL and
excluded. All old science/monitor processes have exited; resident models remain
off during this diagnostic.

S0 exposes a KVM guest with one OS NUMA node, 192 CPUs, and all eight GPUs mapped
to node0. PCI sysfs reports -1 and the single-node OS mapping is recorded as a
fallback, not proof of physical host locality. `numactl -H` is unavailable;
raw sysfs, lscpu, NVIDIA topology and libnuma binding records are retained.
S2 will measure the available node's local 9-MiB H2D and GPU0/GPU1 SHM pair.
Remote H2D and cross-NUMA SHM are unavailable, not simulated or substituted.
This limits NUMA attribution but does not prevent the monitor/affinity stages.

S1 reuses the validated P/CA-rep/Env1 full PLAN and its existing kernel cache.
The CPU-only prefix audit derives the exact 65-layer-event-block prefix
(prefill plus 64 decode forwards), first 64 output tokens and resulting cache
state on all eight ranks. No policy PLAN/model capture is added. Both modes use
identical original node-wide CPU affinity, model, kernel, cache, request and
routing operations. The generation function differs only in horizon constants.
Every fresh process performs one full untimed prefix warmup and validates
hashes before resetting cache keys, request/KV state and RNG.

The HEAVY scan retains the old five-second wait, process-tree RSS/PSS traversal,
resource thresholds and foreign-process checks. In BOUNDARY, all eight workers
announce readiness after warmup/reset; the coordinator completes its final
safety/PSS scan before releasing GO. It then blocks waiting for process exit,
with no resource queries or procfs walks during measurement. Post-run checks
occur only after every worker exits. A 3600-second timeout remains; STOP is
checked at boundaries. The worker uses only existing outer timing boundaries,
no per-token logs/counters, and rejects recompilation or frozen-route/hash drift.
All runs retain host-available/PSS/GPU-memory/temperature admission checks.

S3 fixes eight disjoint visible CPU sets, rank r using CPUs [24r,24r+23], with
strict node0 memory policy from the existing bootstrap. The pageable file-backed
CPU expert pool is unchanged; no new claim about physical host page placement
is made. Native NCCL/CPU threads inherit the fixed affinity before initialization.
The driver follows Env1,Env2,Env2,Env1,Env1,Env2 for decode64. Only if both
independent spreads pass 5% does it confirm both environments with exactly
three decode256 repeats each in the same order. It stops on either failed gate
or execution error and never resumes policy comparisons. No extra sampling,
profiler, controller work, or policy search is introduced.

S2 uses one 9-MiB pinned buffer, 20 warmups/50 copies, verifying its guest NUMA
node with get_mempolicy. The 32/128/512-KiB dispatch-return probe uses 20/50
iterations with fixed local CPU sets. An untimed two-rank preflight validates
SHM/direct/direct, while measured probes have NCCL logging disabled.

Each stage commits/pushes compact receipts. The bounded driver finally restores
guarded resident model workers after scientific children exit, unless its own
STOP marker requests otherwise, and records fresh worker telemetry. The old
matrix STOP marker remains in place. Raw root:
`/home/hwlee/mgo-results/timing_stability_numa_20261004/`.
