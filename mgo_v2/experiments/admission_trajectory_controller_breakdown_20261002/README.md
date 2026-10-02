# Admission trajectory and controller breakdown — 2026-10-02

This study explains **why communication-aware admission wins or loses in the physical runtime**, under the current hardware constraint that only **four GPUs are available for new runs**.

It follows the completed physical locality study at commit `e61758e`. Existing R8 results remain useful retrospective evidence, but **no new R8 job is allowed** in this study.

New GPU work is restricted to physical GPUs **0,1,4,5** and world size **R4**.

The goal is not to invent a new admission policy. First decompose the existing runtime and determine whether the win/loss comes from:

1. immediate communication;
2. future cache/fetch/eviction trajectory;
3. controller implementation cost;
4. rank execution-load skew.

Primary comparison: **Balanced Random vs Hungarian Current**. Same+path is not tuned in this study.
