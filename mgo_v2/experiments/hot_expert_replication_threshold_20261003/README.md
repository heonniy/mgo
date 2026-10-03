# Hot rank-local expert replication threshold

Queued follow-up characterization.

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
