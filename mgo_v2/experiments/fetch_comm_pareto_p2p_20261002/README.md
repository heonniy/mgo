# Fetch/Communication Pareto — H100 P2P on/off

Minimum-scope characterization following the completed controller-overhead packet at `b24ccd6`.

This experiment disables substitution and asks whether expert replication creates a useful trade-off between CPU expert fetch and inter-GPU communication. It compares normal H100 NVSwitch execution with a controlled `NCCL_P2P_DISABLE=1` condition while leaving SHM fallback enabled.

Read `PLAN.md` before implementation. Keep the run matrix bounded; this packet is a go/no-go study, not the final method.
