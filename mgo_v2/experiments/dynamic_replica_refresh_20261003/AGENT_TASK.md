# AGENT TASK — dynamic replica refresh

Checkpoint: d666414.

Read PLAN.md.

1. Hash-verify and reuse the B8/B32 policy-feature traces from d666414.
2. Implement duplicate-to-duplicate refresh without ever evicting a sole global
   copy or active/pinned copy.
3. Reproduce every N0 source cell exactly before any comparison.
4. Run the frozen 120-cell matrix:
   - B32 cache40/60, LRU/GATE, sub OFF/ON, rho .125/.25;
   - B8 cache60, LRU/GATE, sub OFF/ON, rho .125/.25;
   - N0/C1/C2/O4/OR.
5. Report H2D/peer/reloads/locality/unique coverage and replica age/lifetime,
   plus exact refresh resource-price break-evens.
6. Apply the predeclared labels.
7. Commit compact results and stop.

No GPU/model capture, quality eval, E2E/NCCL timing, B64/B128, R8, threshold
retuning or automatic follow-up.
