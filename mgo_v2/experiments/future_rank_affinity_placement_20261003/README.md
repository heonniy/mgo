# Future rank-affinity single-copy placement

This packet follows the completed selective-replication study
(`6a8127c`, `NO_HEADROOM`).

The prior result showed that rank-local demand does recur, but duplicate copies
consume cache slots and are usually evicted before their next useful same-layer
decode opportunity.

This follow-up changes the **action**, not the signal:

> On an unavoidable global miss/reload, keep exactly one copy of the expert and
> choose which currently-demanding rank should own that copy.

No replication, migration or extra speculative fetch is allowed.

See [PLAN.md](PLAN.md), [AGENT_TASK.md](AGENT_TASK.md), and [matrix.json](matrix.json).
