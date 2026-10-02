# Fetch/Communication Pareto — H100 P2P on/off

**Stage 0: BLOCKED_TRANSPORT (2026-10-03).** T0 smoke and three tiny calibration
cells passed. With the requested P2P-disable flag, T1 selected NET/IB/GDRDMA and
failed its first all-to-all. Read [RESULTS.md](RESULTS.md),
[validation.json](validation.json) and [transport_logs/](transport_logs/).
The model matrix has not started; this is not a replica-policy no-go result.
Commit subsequent stage results and failures immediately, per owner request.

Minimum-scope characterization following the completed controller-overhead packet at `b24ccd6`.

This experiment disables substitution and asks whether expert replication creates a useful trade-off between CPU expert fetch and inter-GPU communication. It compares normal H100 NVSwitch execution with a controlled `NCCL_P2P_DISABLE=1` condition while leaving SHM fallback enabled.

Read `PLAN.md` before implementation. Keep the run matrix bounded; this packet is a go/no-go study, not the final method.
