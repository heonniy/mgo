## New priority — Fetch/Communication Pareto with P2P-disabled H100 (2026-10-02)

**Immediate follow-up after `f64eb56`: stable IPC baseline rebase.**
Read `experiments/fetch_comm_pareto_p2p_20261002/IPC_BASELINE_REBASE.md`.
Do not rerun the CPU Pareto screen; its raw H2D/peer-byte results and F/K
selection are transport-independent. Set `NCCL_CUMEM_ENABLE=0` in both new
physical modes. T0 must validate as `P2P/IPC` direct P2P over NVSwitch;
R3 must validate as SHM with P2P disabled and no NET path. After one paired
smoke, re-run the frozen 384-event E1 communication replay. Only STRONG_GAP
may authorize the existing clean F/K stage. Do not use P2P/CUMEM again,
retune NCCL, rerun C or change rho.

**Bounded cuMem diagnosis complete (2026-10-03): CUMEM_PATH_UNSTABLE.**
All three fresh default-T0 trials selected P2P/CUMEM and timed out at the
90-second bound. The single NCCL_CUMEM_ENABLE=0 diagnostic selected P2P/IPC
and passed four-rank payload validation. This supports path-specific
instability, not an established driver root cause. No E1 retry or model
work was started; the alternate path is not adopted as T0. Target GPUs
0,1,4,5 were released; other GPU jobs were not modified. Read
`CUMEM_PREFLIGHT_RETRY_RESULTS.md` and `cumem_preflight_retry.json`.
Stop here: do not add retries, change the timed baseline or reset drivers.

**Immediate follow-up after `b71199c`: bounded P2P/CUMEM diagnosis and conditional E1 retry.**
Read `experiments/fetch_comm_pareto_p2p_20261002/CUMEM_PREFLIGHT_RETRY.md`.
Run exactly three fresh default-T0 32-KiB preflights plus one diagnostic
preflight with `NCCL_CUMEM_ENABLE=0`, each bounded at 90 s. Commit the
diagnosis first. Only if default T0 passes 3/3 on P2P/CUMEM may the original
E1 trace replay be re-run unchanged in a fresh result root. The cuMem-disabled
path is diagnostic only and must not replace T0. If any default trial is
unstable, stop after diagnosis. Do not start E2/model work unless a subsequent
original-path E1 satisfies the existing STRONG_GAP gate.

**E1 trace-communication stage blocked (2026-10-03): BLOCKED_PREFLIGHT.**
The first T0 32-KiB preflight exceeded its 180-second bound. Communicators
initialized and selected P2P/CUMEM; no validated payload receipt followed.
Last logs show shareable-buffer imports/UDS mapping; root cause is not
established. No OOM: peak process-tree RSS 4.69 GiB, minimum host available
1851.92 GiB. Target GPUs were released. The 384-event input/count validation
and two CPU tests pass, but zero trace timings or model runs were executed.
Read `TRACE_COMM_REPLAY_RESULTS.md` and `trace_comm_preflight_failure.json`.
Do not label this NO_GAP: no ratio exists. E2 remains unauthorized by its
STRONG_GAP gate; stop without an automatic retry, NCCL tuning or server move.

**Immediate follow-up after `b6edeb0`: actual-trace comm gate, then conditional clean F/K.**
Read `experiments/fetch_comm_pareto_p2p_20261002/TRACE_COMM_CLEAN_FK.md`.
First replay the recorded 384 decode-layer dispatch/combine traffic with no
model, expert H2D or cache work. Compare T0/R3 in two counter-ordered passes
and commit this E1 result. Only if R3 is slower in both passes and the median
cumulative ratio is >=1.20x may E2 start. E2 removes heavy per-event validation
from the timed path and runs only F=rho0 and K=rho.25, two repeats per
transport (8 cells). Do not rerun C, add rho values, use Nsight, enable
substitution or tune more NCCL knobs. Commit each stage separately and stop.

**Physical F/K/C pilot complete (2026-10-03): NO_CLEAR_SHIFT.**
All six cells passed trace/cache/action/send-count/token parity (2,592 global
layer events). Descriptive winners differ: T0-K (8.3423 s decode) and R3-F
(6.8918 s), with 5.19%/9.30% margins. However, R3/T0 payload intervals are
0.769x at F and 0.866x at K, and F replay/check CPU time differs by 0.991 s;
the proposed communication-cost mechanism is not supported. No repeat or
follow-up is authorized by this result. Read `PHYSICAL_FKC_RESULTS.md` and
`physical_fkc_validation.json` in the packet; stop for owner review.

**Immediate follow-up after `dc7b099`: six-cell physical F/K/C pilot.**
Read `experiments/fetch_comm_pareto_p2p_20261002/PHYSICAL_FKC_PILOT.md`.
Freeze F/K/C=rho 0/.25/.75 CPU action schedules, then physically replay the
same exact-only one-prefill + eight-decode workload on GPUs 0,1,4,5. Run
exactly: T0-F, R3-C, T0-K, R3-K, T0-C, R3-F. One run per cell. Primary
question is whether the descriptively fastest rho moves between T0 and R3.
Require >=5% margin plus consistent H2D/peer counters for PROMISING_SHIFT.
No automatic repeats, longer trace, Nsight, substitution or final method.
Commit after the six cells or immediately on blocking failure and stop.

**CPU-only replica screen complete (2026-10-03): GO_FOR_OWNER_REVIEW.**
All five budgets are nondominated; F/K/C rho=0/.25/.75. F to C increases
decode H2D by 160.20% and reduces peer activation bytes by 100%. All 2,160
CPU events passed, including 432 independent rho=0 checks. No new GPU/model
run. Read `REPLICA_PARETO_RESULTS.md` and `replica_pareto_validation.json`
in the packet. Stop here; do not rerun or start physical validation without
an owner follow-up. The earlier priorities below are retained as history.

**Immediate follow-up after `ed7f82b`: CPU-only replica Pareto screen.**
NCCL characterization is paused. Read
`experiments/fetch_comm_pareto_p2p_20261002/REPLICA_PARETO_SCREEN.md`.
Reuse the validated exact-only 8-decode trace; no new GPU/model run. Sweep
replica-budget rho={0,0.125,0.25,0.5,0.75} with one common deterministic
first-copy rule, LRU, and greedy current-byte-saving replicas. Primary plane is
decode peer activation bytes vs decode expert H2D bytes. Continue only if at
least three points are nondominated and F/C differ by >=10% on both axes.
Commit the CPU result immediately and stop; do not implement the final method
or physical F/K/C automatically.

**Immediate follow-up after `0c09fae`: exact-only payload capture.**
The existing substituted traces cannot recover exact destination ranks. Run one
R4/B8 P0 capture with substitution off, replication off, LRU, T0, and only
1 prefill + 8 decode forwards. Record actual owner maps and dispatch/combine
send counts; derive payload p50/p90/p99/max. Then run only the 32 KiB, 64 KiB,
256 KiB, 1 MiB and 4 MiB T0/R3 crossover with two counter-ordered passes.
Do not implement replication or Stage 1. If R3 is not >=1.5x slower within the
observed exact-only p90/p99 range, stop synthetic H100 P2P-off work and move to
the real no-NVLink server. Commit the bounded result immediately.

Read `experiments/fetch_comm_pareto_p2p_20261002/{README.md,PLAN.md,matrix.json,AGENT_TASK.md}`.

This is the new owner-authorized minimum-scope go/no-go study. Use only GPUs 0,1,4,5; R4/B8/cache30; one prefill + 32 decode forwards; substitution off; LRU fixed. Characterize whether replica budget trades CPU expert H2D against peer activation communication. Compare normal NVSwitch against `NCCL_P2P_DISABLE=1` with SHM fallback left enabled. CPU sweep first; physically validate only F/K/C, two repeats each transport mode (12 primary generations maximum unless the >10% spread rule triggers one targeted repeat). Do not add load balancing, substitution, final weighted objectives, B4/B16 or R8.

**Immediate follow-up after `fad71d2`: payload crossover only.**
Do not start Stage 1/model/replication work. Read
`experiments/fetch_comm_pareto_p2p_20261002/PAYLOAD_CROSSOVER.md`.
Reuse an existing R4/B8 raw-routing trace to compute actual decode rank-pair
payload p50/p90/p99/max, then benchmark only 32 KiB, 64 KiB, 256 KiB, 1 MiB
and 4 MiB per peer under T0 and the validated R3 SHM condition. Two
counter-ordered lightweight passes only. If R3 is not >=1.5x slower within the
real p99 payload range, stop synthetic H100 P2P-off work and move the
communication-sensitive experiment to the real no-NVLink server. Commit the
result immediately; do not tune more NCCL knobs.

**Recovery outcome (2026-10-03): functional, cost-increase gate not met.**
R1 and R2 failed with IB retry errors; R3 passed all-rank payload validation
using `SHM/direct/direct` with P2P_LEVEL=LOC and IB disabled. Stop at that first
success: no loopback retry or further knobs. Its three calibration cells passed,
but peer median was 0.236064 ms versus prior T0 0.336480 ms (0.702x), so there is
no demonstrated communication-cost increase. Stage 1/model/replica work has
not started. Read `TRANSPORT_RECOVERY_RESULTS.md` and
`transport_recovery_result.json` in the packet before further work. The prior
instructions/status below are retained as history; do not automatically rerun
or manufacture a slower condition.

**Immediate transport recovery after `7c881f7`:** do not start the model.
Read `experiments/fetch_comm_pareto_p2p_20261002/TRANSPORT_RECOVERY.md`.
Try only the bounded R1/R2/R3 no-P2P smoke sequence, committing each result.
R1 uses `NCCL_P2P_LEVEL=LOC`; R2 additionally disables GDR; R3 disables IB
and may use Socket/loopback as an explicitly synthetic stress condition. Stop
at the first valid path. If all fail, stop synthetic H100 T1 work and move the
Pareto experiment to the real no-NVLink server rather than exploring more NCCL knobs.

**Stage 0 status (2026-10-03): blocked at T1 transport.** T0 smoke and three
calibration cells passed; T1 selected NET/IB/GDRDMA and failed its first
all-to-all with `IBV_WC_RETRY_EXC_ERR`. See the packet's `RESULTS.md` and
`validation.json`. No model/replica runs have started. Resolve the transport
gate before primary timing; never label this observed path verified SHM.
The owner requests immediate incremental commits at stage boundaries and on
failures, rather than waiting for the whole packet to finish.

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
