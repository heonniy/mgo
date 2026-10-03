# Future rank-affinity single-copy placement

**Complete: NO_PLACEMENT_HEADROOM.** All 18 fixed CPU cells and their
deterministic verification invocations passed. B8 O0 saves 4.9645% peer
bytes, just below the frozen 5% modest gate. Every OH policy increases peer
bytes versus O0, although some reduce H2D. See [RESULTS.md](RESULTS.md)
and [validation.json](validation.json). No GPU run belongs to this packet.

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
