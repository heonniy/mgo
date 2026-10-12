# Induced-contention R8 comparison: NEAR vs FAST vs SPLIT at 1.3x and 1.7x

Environment: same as stage_breakdown_20261012 (R8, serial per-expert executor, inline H2D, prefetch OFF;
Qwen post-generation diagnostic with live tokens, DeepSeek instrumented teacher target). Contention hogs
(`scripts/h2d_contention_hog.py`) are started by the guard on GPUs 0-3 after its foreign-process check:
- L13: back-to-back, 1 stream per GPU -> calibrated copy ratio GPUs0-3/4-7 = 1.29x
- L17: back-to-back, 2 streams per GPU -> 1.72x

Arms (Near placement in all):
- NEAR (`LA_CA_NEAR`): balanced, remainder to GPUs 0-3;
- FAST (`NEAR_FAST`, pcie_quota_r8_b16_20261010/LOOKUP.json): balanced, remainder to GPUs 4-7;
- SPLIT (`NEAR_SPLIT`): Eq.-(6)-style quota for the level's calibrated ratio (SPLIT_r129.json / SPLIT_r172.json).

3 reps; each rep runs both levels and both models with rotated arm order (36 jobs). Metrics as in
stage_breakdown (H2D straggler/imbalance, load imbalance, comm cost, EP cost, TPOT).
