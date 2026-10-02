# Rank-local reuse / selective-replication headroom

This packet asks one narrow question before any new GPU experiment:

> When a rank remotely uses an expert, does the same `(layer, expert, rank)`
> demand recur enough that one additional expert copy can amortize its H2D
> fetch and cache opportunity cost?

The experiment is CPU-only and reuses the existing exact-routing captures.
It does **not** claim a final online policy.

Read [PLAN.md](PLAN.md) and [AGENT_TASK.md](AGENT_TASK.md).
