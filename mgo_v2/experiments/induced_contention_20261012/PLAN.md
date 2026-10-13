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

## Status (2026-10-13)
- Campaign 1 (run.py, STATUS.json): PASS, 36/36 + Qwen smoke. DeepSeek l13 FAST rep 2 and SPLIT rep 3 are whole-run slow outliers
  (comm +20 ms/step, compute imbalance +6 ms/step, H2D wait lower, hogs normal); medians reported alongside means.
- Campaign 2 (run2.py, STATUS2.json): stopped by the owner after 9 of 36 jobs (rep 1: l00 all four arms both models,
  l13 brand Qwen). The interrupted job was kept as icdl13brand_full_v1_STOPPED_by_owner (status FAIL) and is not analyzed.

## Qwen follow-up (run3.py STATUS3.json, run4.py STATUS4.json; 2026-10-13)
MB_BAL (no-hog microbench table), balanced random worst salt 406, balanced random ordinary salts 1/2/3, at 1.3x/1.7x;
plus no-hog ordinary salts 2/3 (salt 1 = campaign 2). n=3 each, ms per decode step.
| level | NEAR | FAST | MB_BAL | BRAND (salt=rep) | BRAND_W (406) | SPLIT |
|---|---|---|---|---|---|---|
| none* H2D imb / TPOT | 10.2 / 433.4 | 6.7 / 428.8 | 6.4 / 427.1 | 8.4 / 430.4 | 8.9 / 432.2 | — |
| 1.3x | 27.0 / 465.1 | 20.5 / 454.4 | 19.7 / 452.6 | 24.4 / 461.0 | 24.9 / 468.0** | 10.8 / 450.3 |
| 1.7x | 74.9 / 543.7 | 65.1 / 525.6 | 64.6 / 524.5 | 71.7 / 535.1 | 72.2 / 539.2 | 30.0 / 484.5 |
* no-hog NEAR/FAST/MB_BAL/BRAND_W from microbench_quota_20261013; BRAND from this packet.
** rep 1 is a whole-run TPOT outlier (478.9 ms with normal H2D imbalance 25.8); median 462.9.
