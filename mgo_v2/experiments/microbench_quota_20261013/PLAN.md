# Qwen R8: microbench-optimal quota and worst-seed balanced random (2026-10-13)

Environment: no hogs; serial per-expert executor, inline H2D, prefetch off, Near placement, live tokens with
post-generation diagnostic; R8 ShareGPT local B16 L512/O64, cache 30%. Code: 08efb78 (origin/paper_result) plus
the local additive commits (hog options off by default; BALANCED_RANDOM_NEAR code 25).

Model (`fluid.py`): per-rank finish times of q_r expert copies when each rank progresses at the microbench rate
for the exact active subset (pcie_all8_concurrency_20261009/RANK_COMBINATIONS.csv, Qwen payload).
- MB_BAL: best balanced row (max-min <= 1) over every choice of remainder ranks.
- MB_OPT: unconstrained, hill-climbed from MB_BAL with single-copy moves. 106/129 rows have max-min > 1
  (up to 7 on rows that occur in the decode); the model lowers the 0-3 group's active count.
- BRAND_W: BALANCED_RANDOM_NEAR with the salt that maximizes the model H2D imbalance over salts 1..2000,
  using the per-event miss counts of icql00near_full_v1 (offline draws match the runtime exactly, 3024/3024).
Arms: NEAR, FAST, MB_BAL, MB_OPT, BRAND_W x 3 reps. RANDOM_QUOTA_NEAR is reused from stage_breakdown_20261012.

## Results (Qwen, n=3, ms per decode step; `analyze.py` -> SUMMARY.json, RUNS_per_step.csv)
| quota | H2D imbalance | EP cost | TPOT | vs NEAR | events max-min>1 |
|---|---|---|---|---|---|
| NEAR | 10.20 ± 0.18 | 159.5 ± 1.0 | 433.4 ± 1.4 | — | 0/9072 |
| FAST | 6.71 ± 0.23 | 147.3 ± 3.0 | 428.8 ± 1.3 | -1.07% | 0/9072 |
| MB_BAL | 6.44 ± 0.19 | 143.6 ± 2.4 | 427.1 ± 1.9 | -1.46% | 0/9072 |
| MB_OPT | 12.27 ± 0.26 | 168.9 ± 3.3 | 440.6 ± 3.2 | +1.66% | 8697/9072 (max 7) |
| BRAND_W (salt 406) | 8.86 ± 0.10 | 136.9 ± 3.7 | 432.2 ± 2.4 | -0.29% | 0/9072 |
| RANDOM_QUOTA (reused) | 38.03 ± 0.28 | 240.4 ± 0.9 | 484.6 ± 0.4 | +11.80% | 9072/9072 (max 20) |
Model (fluid, burst microbench) predicted MB_OPT lowest H2D imbalance (5.6); measured it is the highest of the
balanced-or-near arms: the burst table overstates the in-run asymmetry. Within max-min <= 1, the microbench choice
(MB_BAL) is the best measured arm; the worst balanced-random salt stays near NEAR.
