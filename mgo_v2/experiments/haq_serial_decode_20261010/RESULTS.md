# Stage 2 result (Qwen3): decode miss-placement policies under serial GEMM, no H2D overlap

Hit-aware quota (HAQ) lowers decode TPOT by **3.91% (C50)** and **3.32% (C30)**
versus BW, the worst balanced policy. Both values are medians of 6 unfiltered
targets (2 rounds × 3). The order of the medians is BW > BR > LA_CA_NEAR > HAQ
at both cache sizes. Two controls added later, fully random fetch rank (RANDOM) and static
block placement (STATIC_BLOCK), are slower than BW in both cells; see "Added controls".

## Setting

- Qwen3-30B-A3B-Instruct-2507, R4 on physical GPUs 0/1/4/5 (NVLink), ShareGPT,
  rank-local B16, input 512, output 64. The workloads are the existing frozen
  manifests `qwen_cache_ablation_20261009/WORKLOADS.json` (cells C50 and C30).
  No sample seed was selected.
- Native serial per-expert executor. `--h2d-serial-ablation` makes the host wait
  for all of a layer's demand H2D copies before expert compute, so H2D never
  overlaps compute. Prefetch is off.
- Prefill placement is BR in every run. The decode policy switches at decode
  step 1 (`--decode-policy`).
- One job runs one policy: warmup plus 3 targets, each starting from an empty
  expert cache.
- Round 1 runs BR → LA_CA_NEAR → HAQ → BW. Round 2 runs the reverse order:
  BW → HAQ → LA_CA_NEAR → BR.
- No sample was discarded.

## Policies

All four policies keep the per-rank miss count within ±1 (HAQ caps it at ceil(m/R)).

| Policy | Description |
|---|---|
| BW | Puts the hottest misses on the first maximum-quota rank (worst-case control). |
| BR | Balanced random. |
| LA_CA_NEAR | Existing main_OURS Near policy. |
| HAQ | New policy. Water-fills each rank's miss quota against that rank's resident (hit) expert count, so hit-heavy ranks take fewer misses. Placement within the quota uses the Near rule. See `scripts/haq_placement.py`. |

## Results (TPOT, s/token)

### Qwen3_ShareGPT_R4_C50_B16_L512_O64

| Policy | R1 t1 / t2 / t3 | R2 t1 / t2 / t3 | Median | vs BW | R1 / R2 vs BW |
|---|---|---|---|---|---|
| BW | 0.5241 / 0.5217 / 0.5157 | 0.5213 / 0.5245 / 0.5206 | 0.5215 | +0.00% (+0.0 ms) | +0.00% / +0.00% |
| RANDOM | 0.5513 / 0.5588 / 0.5551 | 0.5533 / 0.5492 / 0.5505 | 0.5523 | +5.90% (+30.8 ms) | +6.40% / +5.60% |
| STATIC_BLOCK | 0.5405 / 0.5373 / 0.5378 | 0.5418 / 0.5362 / 0.5364 | 0.5375 | +3.08% (+16.0 ms) | +3.07% / +2.89% |
| BR | 0.5296 / 0.5160 / 0.5170 | 0.5113 / 0.5106 / 0.5141 | 0.5151 | -1.24% (-6.4 ms) | -0.91% / -1.92% |
| LA_CA_NEAR | 0.5104 / 0.5105 / 0.5074 | 0.5042 / 0.5054 / 0.5034 | 0.5064 | -2.90% (-15.1 ms) | -2.17% / -3.28% |
| HAQ | 0.5016 / 0.5015 / 0.5105 | 0.5007 / 0.5007 / 0.5006 | 0.5011 | -3.91% (-20.4 ms) | -3.86% / -3.96% |

| Policy | Copies/token by rank (R1 t2) | Experts/token by rank (R1 t2) |
|---|---|---|
| BW | [430.1, 417.8, 405.1, 393.8] | [1186.0, 1158.5, 1115.7, 1092.8] |
| RANDOM | [415.9, 405.2, 409.4, 410.1] | [1150.2, 1137.9, 1145.1, 1139.3] |
| STATIC_BLOCK | [401.1, 431.1, 407.6, 387.0] | [1133.4, 1164.2, 1145.7, 1117.9] |
| BR | [422.3, 410.8, 398.3, 385.7] | [1160.2, 1146.8, 1134.5, 1117.9] |
| LA_CA_NEAR | [421.9, 409.7, 397.7, 385.5] | [1163.2, 1142.4, 1127.8, 1123.0] |
| HAQ | [405.1, 406.5, 406.2, 403.7] | [1145.7, 1141.3, 1136.3, 1139.0] |

### Qwen3_ShareGPT_R4_C30_B16_L512_O64

| Policy | R1 t1 / t2 / t3 | R2 t1 / t2 / t3 | Median | vs BW | R1 / R2 vs BW |
|---|---|---|---|---|---|
| BW | 0.5678 / 0.5691 / 0.5679 | 0.5758 / 0.5708 / 0.5753 | 0.5699 | +0.00% (+0.0 ms) | +0.00% / +0.00% |
| RANDOM | 0.6108 / 0.6097 / 0.6129 | 0.6157 / 0.6161 / 0.6173 | 0.6143 | +7.78% (+44.3 ms) | +7.57% / +7.10% |
| STATIC_BLOCK | 0.5840 / 0.5879 / 0.5883 | — (interrupted) | 0.5879 | +3.15% (+17.9 ms) | +3.53% |
| BR | 0.5676 / 0.5678 / 0.5626 | 0.5617 / 0.5623 / 0.5619 | 0.5624 | -1.32% (-7.5 ms) | -0.04% / -2.33% |
| LA_CA_NEAR | 0.5590 / 0.5517 / 0.5530 | 0.5602 / 0.5568 / 0.5569 | 0.5568 | -2.30% (-13.1 ms) | -2.61% / -3.19% |
| HAQ | 0.5499 / 0.5564 / 0.5534 | 0.5486 / 0.5516 / 0.5505 | 0.5510 | -3.32% (-18.9 ms) | -2.54% / -4.31% |

| Policy | Copies/token by rank (R1 t2) | Experts/token by rank (R1 t2) |
|---|---|---|
| BW | [710.3, 698.7, 686.6, 675.0] | [1166.5, 1154.0, 1135.7, 1115.0] |
| RANDOM | [694.1, 682.9, 689.0, 684.7] | [1146.1, 1134.9, 1141.2, 1135.7] |
| STATIC_BLOCK | [692.3, 703.6, 705.9, 655.8] | [1146.5, 1157.3, 1159.7, 1108.5] |
| BR | [703.3, 690.9, 679.2, 667.6] | [1158.7, 1145.9, 1135.3, 1121.7] |
| LA_CA_NEAR | [706.5, 695.3, 682.4, 671.4] | [1161.7, 1149.7, 1138.5, 1125.9] |
| HAQ | [691.0, 690.0, 688.0, 686.3] | [1146.6, 1145.0, 1143.7, 1140.3] |

Copies and experts are per decode token for each rank (rank 0→3), read from the
runtime counters.

## Interpretation

Serial GEMM costs roughly 0.2 ms per expert in this runtime (GEMM launch plus
return combine). With no overlap, the critical path of each layer is the
busiest rank's (copies + experts).

- **BR and LA_CA_NEAR:** the balanced quota always gives the +1 remainder copy
  to the lowest rank first. This leaves a rank 0 → 3 copy gradient.
- **BW:** concentrates hot misses on rank 0. It also adds about 30 copies per
  token, from slot contention on that rank.
- **HAQ:** flattens both copies and experts across ranks without raising the
  total copy count.

A CPU replay of the production controller on the ShareGPT route pool predicted
the same order. With c ≈ 0.2 ms per expert, the predicted HAQ gain over BR is
−14.7 ms/token, against −15.4 ms/token measured at C50.

## Added controls (2026-10-11)

Two controls were run later, with the same flags and the same manifests as the
original jobs.

- **RANDOM (code 19).** Every miss goes to a uniformly random rank, with no
  quota. `MGO_RANDOM_SALT` is set to the round number, so the two rounds use
  different random streams.
- **STATIC_BLOCK (code 18).** Every miss goes to rank `expert_id // (E/R)`.

Both controls are slower than BW in every cell.

- **RANDOM is +5.9% (C50) and +7.8% (C30).** Without a quota, some layers pile
  copies onto one rank, even though the per-token totals in the table look
  balanced.
- **STATIC_BLOCK is +3.1% (C50) and +3.2% (C30).** Its static owner is
  unbalanced from layer to layer.

Measured against RANDOM, HAQ cuts TPOT by 9.3% (C50) and 10.3% (C30).

**The C30 STATIC_BLOCK round 2 job was interrupted by the owner after one
target.** Its partial receipts are kept in `raw_interrupted/` and are excluded
from the tables. The C30 STATIC_BLOCK row therefore uses the 3 targets of
round 1 only.

`run_static2.sh` and `run_random.sh` hold the exact commands.

## Caveats

- **Generated tokens differ between policies, so routes diverge.** The
  rank-partial BF16 combine order depends on placement. Within each policy, all
  6 targets have an identical `argmax_hash`. This is an end-to-end comparison
  on the same prompts, not a fixed-route comparison.
- **Run-to-run spread is about 1% within a policy.** For example, BW at C30 is
  0.5679 in round 1 and 0.5753 in round 2. The HAQ vs LA_CA_NEAR gap (about 1%)
  is within that spread. The HAQ vs BW gap (3.3–3.9%) is not.
- **This is the serial, non-overlapped ablation mode.** In the default grouped
  GEMM + overlap runtime, policy differences stay around 1%.
- **The HAQ cost constants are a first guess.** The code uses 180 µs per copy and
  71 µs per expert. These were not tuned to the ~0.2 ms effective expert cost
  measured afterwards.

## Reproduce

`run_matrix2.sh` holds the exact job sequence. The runner is
`scripts/run_headline_job.py` with `--decode-policy`. `python summarize.py`
rebuilds `SUMMARY.json` and the tables from `raw/`. `raw/<cell>_<policy>_r<round>/`
holds each job's per-rank receipts, `status.json` (exact command and source
commit), and `run.log`.
