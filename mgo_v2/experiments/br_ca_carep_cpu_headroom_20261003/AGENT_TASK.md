# AGENT TASK — BR / CA / CA-rep CPU headroom

Checkpoint: e97f153.

This owner request supersedes further dynamic stale-replica-refresh exploration.
If a refresh process is already running, let the current bounded process/cell
finish and preserve its checkpoint, then do not launch more refresh cells.

Read PLAN.md.

1. Freeze dataset manifests:
   - MATH train: 128 calibration examples, disjoint, seed 43;
   - MATH test: 512 workload examples, seed 42;
   - stratified by subject/category and difficulty.
2. In one 8-GPU model session:
   - calibrate co-routed expert-output cosine similarity;
   - extend calibration once to 256 only if the frozen coverage gate fails;
   - capture one 512-request, fixed-64-decode master routing trace.
3. Repack that master trace offline into every R={4,8},
   B={8,16,32,64} workload.
4. Use global cache ratios {30,40,50,60}% independent of R and eviction
   {LRU,Gate}; substitution OFF/ON when calibration is valid.
5. Implement/validate:
   - BR balanced random seed42;
   - CA exact current-demand balanced oracle;
   - CA-rep = CA + one future-popularity replica when optimistic remaining
     same-layer peer saving >=9 MiB; replica remains normally evictable.
6. Run at most 384 main CPU replays + 16 BR seed-audit replays.
7. Report exact/global/local/substitute/effective hits, residual misses,
   turnover/reloads, H2D/peer/locality, and BR->CA / CA->CA-rep headroom.
8. Apply frozen labels and commit compact results.
9. Stop. No Env1/Env2 timing, quality eval, NCCL, B128, Coverage eviction,
   online controller or replica protection.

Do not create one GPU trace per matrix cell.
