# Stage-wise cost breakdown (R8, B16, C30, serial GEMM + inline H2D)

Campaign: `run.py`, 5 arms x 2 models, rotated orders. Reps 1-3 completed (30/40 jobs); the runner then
stopped at the rep-4 STOP check (a STOP file appeared and was later removed by its creator), so every
cell has n=3. Aggregation: `analyze.py` -> `SUMMARY.json`; figure: `plot.py` -> `figure_a.png`.
All quantities are decode-only sums over layers, seconds per decode run, mean +- std over 3 runs.

## (a) Stage 1 quota: H2D straggler (sum_l max_r H2D wait)

| Model | NEAR quota | FAST quota | change | mean wait NEAR / FAST |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 3.922 +- 0.010 | 3.653 +- 0.057 | -6.9% | 3.267 / 3.211 |
| DeepSeek-V2-Lite | 3.163 +- 0.015 | 2.881 +- 0.002 | -8.9% | 2.616 / 2.613 |

FAST does not reduce the average copy wait; it shrinks the slowest rank's wait.

## (b) Stage 2 placement: EP cost = rank imbalance (H2D wait + compute) + pure comm

| Model | Arm | rank imbalance | compute-only imbalance | pure comm | EP cost | TPOT vs NEAR |
|---|---|---:|---:|---:|---:|---:|
| Qwen | FAST+Worst | 6.607 | 6.475 | 10.861 | 17.469 +- 0.108 | +21.1% |
| Qwen | FAST+Random | 2.332 | 2.203 | 8.454 | 10.786 +- 0.031 | +1.7% |
| Qwen | FAST+Near | 1.473 | 1.314 | 7.956 | 9.429 +- 0.192 | -0.7% |
| Qwen | HAQ_FAST | 1.086 | 1.050 | 7.581 | 8.667 +- 0.116 | -1.6% |
| DeepSeek | FAST+Worst | 1.785 | 1.735 | 4.351 | 6.135 +- 0.136 | +6.5% |
| DeepSeek | FAST+Random | 0.992 | 0.899 | 4.030 | 5.022 +- 0.071 | -0.4% |
| DeepSeek | FAST+Near | 0.825 | 0.707 | 3.990 | 4.815 +- 0.007 | -1.9% |
| DeepSeek | HAQ_FAST | 0.673 | 0.611 | 3.965 | 4.638 +- 0.040 | -3.5% |

EP cost equals the per-rank mean of collective time (a2a + gather), which includes waiting for the
busiest rank; "pure comm" subtracts that wait. TPOT: Qwen from the uninstrumented timed target,
DeepSeek from the instrumented teacher-token target (per-rep values in SUMMARY.json; std <= 0.004 s).

Findings:
- Ordering Worst > Random > Near > HAQ_FAST holds in both models, for both imbalance and pure comm.
- HAQ_FAST is the best arm in serial mode: its hit-aware quota balances hit+miss compute per rank
  (compute imbalance -20% Qwen, -14% DeepSeek vs FAST+Near) at a small H2D-straggler cost (+1-2%).
  In grouped mode (launch-bound GEMM) this term had nothing to optimize, so HAQ_FAST == FAST there.
