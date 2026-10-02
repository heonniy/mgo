# Fetch/Communication Pareto — H100 P2P on/off

**Current recovery outcome: FUNCTIONAL_NO_COST_INCREASE.** R3 passed via
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
