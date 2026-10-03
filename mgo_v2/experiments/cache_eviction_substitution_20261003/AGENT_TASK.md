# AGENT TASK — cache × eviction × substitution

Checkpoint: 6fadd3a.

Read PLAN.md.

1. Audit whether existing B8/B32 captures contain enough gate-history state.
2. If not, make exactly two diagnostic exact captures (B8 and B32) that store
   per-event 128-element GateHistory score vectors, not full router tensors.
   Verify tokens/routes against the existing captures.
3. Implement one replay combining variable cache size, LRU/GATE/COVERAGE,
   substitution OFF/ON, and historical greedy replica admission.
4. First reproduce the five historical cache30/LRU/exact B8 rho points exactly.
5. Run the 120-cell B8 grid, 24-cell fixed-460 B8 control, and 120-cell B32
   grid: 264 CPU cells total.
6. Report H2D/peer frontiers, reloads, unique coverage, replica lifetime/reuse,
   substitution rates/gate mass/similarity/tier breakdown, and lambda_first.
7. Apply the frozen interpretation labels and commit compact results.
8. Stop. No quality run, B64/B128, NCCL timing, R8 or physical F/K/C follows
   automatically.

Substitution parameters stay gate=.20 and similarity=.65. Coverage stays
W=128, k=1, lambda=2. Do not tune from outcomes.
