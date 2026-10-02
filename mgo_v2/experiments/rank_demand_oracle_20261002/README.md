# Rank-demand oracle placement and GPU critical-path validation — 2026-10-02

This follow-up answers two questions raised by the completed R4 trajectory study at commit `86fc58a`:

1. does communication-aware placement really increase **actual max-rank expert GPU compute time**, not just the expert-row proxy?
2. if placement ignores communication and instead uses exact current routed demand to balance expert work across ranks, how much TPOT/E2E headroom is available?

Only four GPUs are available. Every new physical run uses GPUs **0,1,4,5** as R4.

The new policy is an **oracle diagnostic**, not a deployable method. It uses exact current-event expert demand and an exact min-max assignment to minimize the busiest rank's predicted expert GEMM rows. Oracle solve time is measured separately and excluded from the frozen-plan E2E replay so the result represents data-plane headroom.

No communication term, same-layer affinity, path affinity, future-demand predictor, replication, or migration is used by the oracle.
