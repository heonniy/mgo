> **Repetition override:** `REPETITION_AMENDMENT.md` supersedes all earlier three-repeat / two-extra-repeat instructions below. Use two clean samples, at most one conditional third, independently per environment/policy.

# Frozen physical execution

Owner commit: ead25216ddedca87ecdf30eb8f5491cdeca92d14.

Exactly the selected ShareGPT/R8/B8/cache30 manifest is used, preserving all
64 request IDs and their rank-local order. Cache has 1843 slots across eight
ranks. Gate W128, substitution OFF, BR42 and deterministic CA remain unchanged.
The full256 CPU compatibility gate compares incremental physical PLAN policy
with the previously committed exact replay, including final state hashes.

Reuse the physical expert storage, real H2D, NCCL exchange, bounded cache,
expert kernels and schedule tensor layout from env_offload_worker. Reuse
`timing_stability_residency_worker.generate` with NORMAL residency and no fault
recording; this keeps its globally synchronized timing boundaries. The same
fixed, disjoint 24-vCPU rank affinity is applied before model loading. No claim
about physical host NUMA is made from the guest topology.

Two untimed transport preflights require Env1 P2P/IPC and Env2 SHM/direct/direct.
Both use NCCL_CUMEM_ENABLE=0; Env2 additionally uses NCCL_P2P_DISABLE=1 and
NCCL_IB_DISABLE=1. No NCCL debug logging is enabled in model timing processes.

Generate one live physical PLAN per policy in Env1 and reuse it in both
environments. This freezes the physical route/action/token/cache trajectory.
A CPU structural audit verifies schedule bytes, routes, transfers, slots and
final state. Record route/token differences from the original native capture
in PLAN_CPU_comparison.json rather than assuming cross-implementation bit
parity. COMPILATION/WARMUP, each MEASURE repeat and COUNTERS must reproduce
their own frozen PLAN exactly. COUNTERS must also match across environments;
compare actual peer/H2D bytes explicitly to the archived CPU expectation.

Run one separate COMPILE per policy/environment. Every MEASURE then uses a
fresh process and full schedule warmup, resets cache/KV/RNG, signals readiness,
and waits for the coordinator's completed boundary safety check. From GO until
process exit, the coordinator waits without resource polling, PSS/smaps walks,
GPU queries or profiling. The runtime forbids recompilation and executes only
frozen actions; controller/solver work is outside timing. PLAN and COUNTERS are
untimed, separate processes. Per-rank receipt hashes and actual transfer bytes
are retained outside Git, with compact phase/validation receipts committed.

The twelve primary MEASURE runs use exactly the order in PLAN.md. The 5%
spread gate is (max-min)/median, evaluated for E2E, decode wall and TPOT in each
policy/environment. If any exceeds 5%, run two additional repeats per policy
only in that environment: repeat4 CA then BR; repeat5 BR then CA. Never add
more than these planned repeats. Report all three or five samples and mark
remaining instability without claiming a reliable performance ordering.

Safety: stop only project-owned resident workers, require all eight GPUs free,
>=768 GiB host memory available, >=76000 MiB free per GPU and temperature <65C
before a phase. Untimed safety polling requires host available >=256 GiB,
GPU free >8192 MiB and temperature <85C and no foreign GPU process. Timed phase
is bounded to one hour after GO, all phases to two hours. On errors stop/reap
only this phase's process group, preserve results and restore owned model
workers after scientific processes exit. The original paused matrix remains
paused. Commit/push each phase and the final result. No followup sweep.
