# mgo_v2 coding instructions

## New priority — exact rank-demand oracle and GPU critical-path validation (2026-10-02)

Read `experiments/rank_demand_oracle_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

**Owner scope reduction (2026-10-02):** the latest user instruction requests the
minimum sufficient experiment set. Read `scope_amendment.json` in that packet.
Preserve the completed six-repeat B4 results; B8/B16 use two repeats per policy
(30 primary generations total). Controller follow-up is B8 only, one fresh
C0/C1/C2 generation per policy (9 total), full decision/cache/token parity and
separate diagnostics. Do not expand controller timing to B4/B16. This amendment
overrides the larger repetition/expansion matrix in the earlier plan.

Use only physical GPUs **0,1,4,5** as R4. Compare Balanced Random, existing Hungarian-current, and an exact diagnostic **rank-demand oracle** that ignores communication and minimizes the busiest rank's current effective expert-token rows under the same hard admission-count quota. Oracle assignments are generated in an untimed planning pass and replayed frozen for primary E2E/TPOT, with exact plan/cache/token parity required; solver time is reported separately. Run B4/B8/B16, 64 decode forwards, six uninstrumented repetitions using all policy-order permutations once per batch. After timing, profile only R4/B8 to measure actual max-rank expert GPU time, NCCL, H2D and layer completion. Do not add a joint communication+load policy before this headroom study is complete.

**Oracle packet status: complete.** The owner-reduced 30 generations and three full B8 profiles passed at `experiments/rank_demand_oracle_20261002/{RESULTS.md,validation.json}`. Do not rerun the packet. Continue only the separately authorized reduced controller-overhead follow-up.

**Controller follow-up status: complete.** The reduced B8 packet passed at
`experiments/controller_overhead_20261002/{RESULTS.md,validation.json}`: nine
primary generations, 9,360 captured CPU differential events and two short P1
profiles. C1/C2 remain opt-in. Do not automatically rerun, add repetitions,
expand to B4/B16 or change placement policies. Single-sample timing is descriptive.

## New priority — admission trajectory/controller breakdown (2026-10-02)

The physical locality study is complete at `e61758e`. Before changing the method, read `experiments/admission_trajectory_controller_breakdown_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

**Status: the bounded trajectory packet is complete.** Read `experiments/admission_trajectory_controller_breakdown_20261002/{RESULTS.md,validation.json,implementation_audit.md}`. Six R4 diagnostics and 60 CPU replays passed; do not automatically rerun the packet. Any method change or controller optimization remains a separate owner-reviewed follow-up.

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
