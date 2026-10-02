# Rank-local reuse / selective-replication headroom

**Complete — NO_HEADROOM (CPU accounting).** B8 reproduces F exactly and
passes the reuse gate (73.70% of marginal bytes recur within four steps),
but none of the 16 selective policies dominates an old nonzero-rho point.
B16/B32 secondary traces are also complete: 48 fixed CPU cells in total.
Peak RSS was 720.78 MiB; the eight resident-model GPU workers were unchanged.

Read [RESULTS.md](RESULTS.md), [validation.json](validation.json), and
[execution conventions](EXECUTION_PROTOCOL.md). The bounded study is finished;
no additional GPU experiment or online controller follows automatically.

This packet asks one narrow question before any new GPU experiment:

> When a rank remotely uses an expert, does the same `(layer, expert, rank)`
> demand recur enough that one additional expert copy can amortize its H2D
> fetch and cache opportunity cost?

The experiment is CPU-only and reuses the existing exact-routing captures.
It does **not** claim a final online policy.

Read [PLAN.md](PLAN.md) and [AGENT_TASK.md](AGENT_TASK.md).
