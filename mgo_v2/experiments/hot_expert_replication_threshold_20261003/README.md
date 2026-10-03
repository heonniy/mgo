# Hot rank-local expert replication threshold

**Complete: CURRENT_BATCH_TOO_COLD under the frozen pooled-median gate.**
The concurrent-H2D thresholds are T0 >512 and R3 512, versus observed B32
maximum remote demand 32. H3 and larger-batch capture are skipped. R3's
pass-dependent crossovers and single-rank p90 n=1 sensitivity prevent a
robust hardware-threshold claim. Read [RESULTS.md](RESULTS.md) and
[validation.json](validation.json); no extra runs follow automatically.

Question:

> Is there a rank-local token-demand threshold above which paying one expert
> H2D fetch to create a local copy is cheaper than repeatedly serving those
> token routes remotely?

This is the "hot expert local, cold expert remote" version of the original
slow-interconnect replication hypothesis.

This packet is **queued**. It must not preempt the currently active
future-rank-affinity single-copy placement packet.

Read [PLAN.md](PLAN.md), [AGENT_TASK.md](AGENT_TASK.md), and
[matrix.json](matrix.json).
