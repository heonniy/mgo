# Rank-demand oracle placement and GPU critical-path validation — 2026-10-02

**Status: complete under the owner-reduced scope.** Read [RESULTS.md](RESULTS.md), [validation.json](validation.json) and [scope_amendment.json](scope_amendment.json). All 30 primary generations, three full B8 profiles and 9,360 exact planning events passed. The original measured controller/runtime remain unchanged; controller repair is recorded in the separate `controller_overhead_20261002` packet.

This follow-up answers two questions raised by the completed R4 trajectory study at commit `86fc58a`:

1. does communication-aware placement really increase **actual max-rank expert GPU compute time**, not just the expert-row proxy?
2. if placement ignores communication and instead uses exact current routed demand to balance expert work across ranks, how much TPOT/E2E headroom is available?

Only four GPUs are available. Every new physical run uses GPUs **0,1,4,5** as R4.

The new policy is an **oracle diagnostic**, not a deployable method. It uses exact current-event expert demand and an exact min-max assignment to minimize the busiest rank's predicted expert GEMM rows. Oracle solve time is measured separately and excluded from the frozen-plan E2E replay so the result represents data-plane headroom.

No communication term, same-layer affinity, path affinity, future-demand predictor, replication, or migration is used by the oracle.


## Controller-overhead follow-up

The placement/oracle experiment remains unchanged. After its result packet is complete, run the isolated controller-overhead repair stage in `PLAN.md`: optimize decision-equivalent Coverage/cache bookkeeping first, then test single-planner broadcast versus replicated per-rank planning. Controller changes must preserve event-level plans, victims, cache state and generated tokens, and their TPOT/E2E impact must be reported separately from placement-policy gains.
