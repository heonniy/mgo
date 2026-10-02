# mgo_v2 coding instructions

## New priority — PCIe/NUMA fabric characterization on shared 4-GPU server (2026-10-02)

Read `experiments/pcie_numa_fabric_characterization_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

Use exactly physical GPUs **1,2,3,5** with strict NUMA-local host fetch: GPU1/2 -> NUMA0 and GPU3/5 -> NUMA1. The selected topology has same-NUMA PXB pairs (1,2) and (3,5), while cross-group pairs are SYS. Measure 9 MiB expert H2D contention, same-NUMA versus cross-NUMA GPU communication, four-rank NCCL, and H2D+communication co-contention. The owner permits short overlap with unrelated shared-server GPU jobs; never stop or modify those jobs, and record background utilization so results are labeled contention-inclusive rather than isolated peak bandwidth. Do not launch topology-aware policy tuning from this stage.

## New priority — admission trajectory/controller breakdown (2026-10-02)

The physical locality study is complete at `e61758e`. Before changing the method, read `experiments/admission_trajectory_controller_breakdown_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

Only four GPUs are available for new work. Add diagnostic-only subcomponent timers/counters to `plan_layer`, capture cache/fetch/eviction trajectories, and run matched-raw-demand controller replay. GPU scope is exactly six **R4** diagnostic runs: Random/Current at local B4, B8 and B16, always on physical GPUs 0,1,4,5. **Do not launch any R8 job.** Existing R8 results at `e61758e` are retrospective context only. Existing five-repeat E2E results remain the performance evidence. Do not tune or redesign admission, optimize controller code, add token-load constraints, or revive same+path before this mechanistic packet is complete.

## New priority — local/remote TPOT and E2E impact study (2026-10-01)

Read `experiments/local_remote_e2e_impact_20261001/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}` before new timing work.

Use the validated mgo_v2 physical offloading runtime. Stage A isolates peer-communication sensitivity with resident experts and zero H2D inside timed ranges. Stage B measures uninstrumented five-repeat TPOT/E2E for Random vs Hungarian-current vs Hungarian-same+path at R4/R8 cache30 with 64 fixed decode steps. Freeze same-layer support64/alpha=.25 and path support64/eta=.5 explicitly; do not use the historical alpha=1 default. Profile only after uninstrumented timing and do not retune from timing outcomes.

The mgo_v2 directory is the current implementation target.

## Authority

- mgo_v2 GlobalExpertController is the only logical cache authority.
- Do not re-enable legacy DeviceMapManager, ExpertPrefetcher cache replacement,
  ExpertCache eviction, or Archer autonomous sparse eviction.
- Preserve no-replication/no-migration semantics unless an experiment
  explicitly changes them.

## Frozen research defaults

- expert-level substitution;
- gate protection threshold .20;
- similarity threshold .65;
- Gate history W=128;
- diversity eviction k=1, lambda=2;
- hard-balanced admission quota;
- no CPU attention.

## Correctness before performance

Order of work:

1. CPU policy tests;
2. exact R1/R4 EP parity;
3. cache state parity;
4. substitution parity;
5. eviction parity;
6. admission parity;
7. R8;
8. low-level overlap and D2D optimization;
9. final timing.

Never report performance from a run that still uses the old RPC expert path or
has two residency controllers active.
