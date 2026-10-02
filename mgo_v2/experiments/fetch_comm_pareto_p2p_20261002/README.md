# Fetch/Communication Pareto — H100 P2P on/off

**Current diagnosis: CUMEM_PATH_UNSTABLE (2026-10-03).**
Default P2P/CUMEM timed out in all three 90-second trials. The one
cuMem-disabled diagnostic passed on P2P/IPC. This does not authorize
adopting IPC as the T0 baseline. E1 was not retried and no model was started.
Read [CUMEM_PREFLIGHT_RETRY_RESULTS.md](CUMEM_PREFLIGHT_RETRY_RESULTS.md),
[diagnosis CSV](cumem_preflight_retry.csv), and
[full evidence](cumem_preflight_retry.json). Target GPUs are released;
stop for owner review.

**Prior status: E1 BLOCKED_PREFLIGHT (2026-10-03).**
T0 selected P2P/CUMEM but its tiny preflight did not complete within the
180-second bound. No trace timing or E2 model run was started. This is not
a NO_GAP measurement. The target GPUs were released without OOM. Read
[TRACE_COMM_REPLAY_RESULTS.md](TRACE_COMM_REPLAY_RESULTS.md),
[failure receipt](trace_comm_preflight_failure.json), and
[stage JSON](trace_comm_replay.json). Preserve the failure and stop for review.

**Prior result: physical pilot NO_CLEAR_SHIFT (2026-10-03).**
All six cells passed correctness. T0-K and R3-F are descriptively fastest,
but the communication-cost mechanism is not supported and application/check
CPU timing varies substantially. No repeats were added; GPUs 0,1,4,5 were
released after completion. Read [PHYSICAL_FKC_RESULTS.md](PHYSICAL_FKC_RESULTS.md),
[physical CSV](physical_fkc_pilot.csv), [full result](physical_fkc_pilot.json),
and [validation](physical_fkc_validation.json). Stop for owner review.

**Prior result: CPU replica screen GO_FOR_OWNER_REVIEW (2026-10-03).**
All five rho settings are nondominated. F/K/C are rho=0/.25/.75; from F to C,
decode H2D increases 160.20% and peer activation bytes fall 100%. This reused
the exact-only eight-decode trace with no new GPU/model run. All 2,160 CPU
events passed; independent rho=0 parity passed at all 432 events. Peak CPU
RSS was 201.88 MiB and the sweep took 98.22 seconds. Stop for owner review.

Read [REPLICA_PARETO_RESULTS.md](REPLICA_PARETO_RESULTS.md),
[summary CSV](replica_pareto_screen.csv), [full JSON](replica_pareto_screen.json),
[validation](replica_pareto_validation.json), and
[frozen replay protocol](REPLICA_REPLAY_PROTOCOL.md).
The earlier transport/recovery statuses below are historical.

**Historical recovery outcome: FUNCTIONAL_NO_COST_INCREASE.** R3 passed via
`SHM/direct/direct` with IB disabled. Its peer median was 0.236064 ms versus
T0 0.336480 ms (0.702x). The required cost-increase gate for Stage 1 is not met;
no model/replica work has started. R1/R2 failed; R3 was the first success, so no
loopback retry or additional transport tuning was run.

For the authorized recovery after the original failure, read
[TRANSPORT_RECOVERY_RESULTS.md](TRANSPORT_RECOVERY_RESULTS.md) and
[transport_recovery_result.json](transport_recovery_result.json).
Each R-condition is committed before the next attempt; the original Stage 0
result below is retained as history.

**Stage 0: BLOCKED_TRANSPORT (2026-10-03).** T0 smoke and three tiny calibration
cells passed. With the requested P2P-disable flag, T1 selected NET/IB/GDRDMA and
failed its first all-to-all. Read [RESULTS.md](RESULTS.md),
[validation.json](validation.json) and [transport_logs/](transport_logs/).
The model matrix has not started; this is not a replica-policy no-go result.
Commit subsequent stage results and failures immediately, per owner request.

Minimum-scope characterization following the completed controller-overhead packet at `b24ccd6`.

This experiment disables substitution and asks whether expert replication creates a useful trade-off between CPU expert fetch and inter-GPU communication. It compares normal H100 NVSwitch execution with a controlled `NCCL_P2P_DISABLE=1` condition while leaving SHM fallback enabled.

Read `PLAN.md` before implementation. Keep the run matrix bounded; this packet is a go/no-go study, not the final method.
