# Admission trajectory and controller breakdown — 2026-10-02

This study explains **why communication-aware admission wins in some physical runs and loses in others**.

It follows the completed physical locality study at commit `e61758e`. That study established:

- Hungarian-current reduces remote token-rank pairs in every measured cell;
- R8/B4 shows a large E2E/TPOT improvement;
- R8/B8 and R4/B8 regress;
- the measured `controller_seconds` tracks the sign of the E2E change much more closely than remote-pair reduction alone.

The goal here is not to invent a new admission policy. First decompose the existing runtime and determine whether the win/loss comes from:

1. immediate communication;
2. future cache/fetch/eviction trajectory;
3. controller implementation cost;
4. rank execution-load skew.

Only after those causes are measured should a new policy be proposed.

Primary comparison: **Balanced Random vs Hungarian Current**. Same+path is not tuned in this study.
