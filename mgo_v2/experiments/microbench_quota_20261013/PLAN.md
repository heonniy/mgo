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
