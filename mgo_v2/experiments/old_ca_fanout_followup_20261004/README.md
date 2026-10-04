# Old-CA token-to-rank fan-out follow-up

Queued after the active B32/decode64 tolerance_001 physical packet fully exits
and commits. Do not interrupt that work.

Purpose: explain why the original mgo runtime showed TPOT gains while the clean
current CA can reduce peer bytes without reducing TPOT.

This follow-up isolates one policy difference:

- BR: balanced random miss placement.
- Current-CA: balanced Hungarian placement maximizing current expert-route
  locality.
- OldCA-fanout: the original a0be82e-style balanced Hungarian objective that
  minimizes incremental **token -> remote-rank destinations** for incoming
  experts.

All three policies use the same frozen routes, B32/decode64/cache30/Gate W128,
substitution OFF, single-copy experts and the current clean physical runtime.
No affinity/path bonus is enabled in this first isolation.

The primary question is whether reducing token fan-out, rather than aggregate
peer bytes alone, produces a larger TPOT effect.
