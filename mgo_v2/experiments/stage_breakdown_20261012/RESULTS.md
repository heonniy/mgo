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

---

## Panel view: H2D imbalance, compute/comm panels and TPOT ladder (added by a second session)

Same SUMMARY.json. Figure `stage_breakdown.png` / `.pdf` from `figure.py`: rows = model; panels (a) H2D straggler imbalance NEAR vs FAST, (b) expert-compute imbalance, (c) comm cost for the FAST-quota placements and HAQ_FAST.

### (1) Stage 1 quota: NEAR -> FAST

| Model | Arm | H2D straggler | H2D imbalance | TPOT | vs NEAR |
|---|---|---:|---:|---:|---:|
| Qwen3 | NEAR | 3.92 ± 0.01 | 0.65 ± 0.01 | 0.4333 | 0 |
| Qwen3 | FAST | 3.65 ± 0.06 (-7%) | 0.44 ± 0.01 (**-33%**) | 0.4303 | -0.7% |
| DeepSeek | NEAR | 3.16 ± 0.02 | 0.55 ± 0.00 | 0.2621 | 0 |
| DeepSeek | FAST | 2.88 ± 0.00 (-9%) | 0.27 ± 0.01 (**-51%**) | 0.2572 | -1.9% |

### (2) Stage 2 placement on the FAST quota (plus HAQ_FAST)

| Model | Arm | compute imbalance | comm cost | EP cost | EP vs FAST+Near | TPOT | vs FAST+Near |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | FAST+Worst | 6.47 ± 0.01 | 10.86 ± 0.07 | 17.47 ± 0.11 | +85% | 0.5245 | +21.9% |
| Qwen3 | FAST+Random | 2.20 ± 0.00 | 8.45 ± 0.04 | 10.79 ± 0.03 | +14% | 0.4407 | +2.4% |
| Qwen3 | FAST+Near | 1.31 ± 0.04 | 7.96 ± 0.13 | 9.43 ± 0.19 | 0 | 0.4303 | 0 |
| Qwen3 | HAQ_FAST | **1.05 ± 0.02** | **7.58 ± 0.09** | **8.67 ± 0.12** | **-8%** | 0.4264 | -0.9% |
| DeepSeek | FAST+Worst | 1.73 ± 0.03 | 4.35 ± 0.11 | 6.14 ± 0.14 | +27% | 0.2791 | +8.5% |
| DeepSeek | FAST+Random | 0.90 ± 0.02 | 4.03 ± 0.06 | 5.02 ± 0.07 | +4% | 0.2610 | +1.5% |
| DeepSeek | FAST+Near | 0.71 ± 0.01 | 3.99 ± 0.01 | 4.81 ± 0.01 | 0 | 0.2572 | 0 |
| DeepSeek | HAQ_FAST | **0.61 ± 0.02** | 3.96 ± 0.02 | **4.64 ± 0.04** | **-4%** | 0.2529 | -1.7% |

HAQ_FAST keeps the stage-1 H2D benefit: H2D imbalance Qwen 0.46 (FAST 0.44, NEAR 0.65), DeepSeek
0.32 (FAST 0.27, NEAR 0.55).

### TPOT ladder versus NEAR

| Step | Qwen3 | DeepSeek |
|---|---:|---:|
| NEAR -> FAST (quota / H2D) | -0.7% | -1.9% |
| FAST -> HAQ_FAST (placement / EP) | -0.9% | -1.7% |
| NEAR -> HAQ_FAST | **-1.6%** | **-3.5%** |
| FAST+Worst (bad placement) | +21.1% | +6.5% |

Notes:
- Qwen TPOT is the uninstrumented timed target. DeepSeek TPOT is the instrumented teacher target,
  so its absolute value is inflated, but arms are comparable.
- Qwen diagnostics use live tokens (post-generation pass); DeepSeek uses the frozen NEAR teacher.
- HAQ_FAST's copy-cost constants (250/230 us) were calibrated on Qwen 9 MiB experts and reused for DeepSeek.
- The breakdown is span-based: comm cost is net of the wait for the busiest rank. Read the relative
  changes, not exact wire time.

Two-column variant: `stage_breakdown_ep.png` / `.pdf` (`figure_ep.py`). (a) H2D imbalance NEAR vs FAST;
(b) EP cost as one stacked bar = load imbalance (compute + H2D wait) + communication cost.

Per-round raw values: `RUNS.csv`, one row per measured run (model, arm, policy, round, job), in decode-sum
seconds: h2d_straggler, h2d_imbalance, mean_wait, mean_compute, compute_imbalance, own_imbalance
(compute + H2D wait), comm_cost, ep_cost (= own_imbalance + comm_cost), and TPOT (s/token). The cell means
and stds in the tables above are the mean/stdev of these rows. Two-column figure (b) now shows only
FAST+Random and FAST+Near (same FAST quota).

Per-decode-step versions (decode sums / 63 steps, in ms; TPOT in ms): `RUNS_per_step.csv` (per round) and
`SUMMARY_per_step.csv` (mean/std per cell). `stage_breakdown_ep.png` is now plotted in ms per decode step.
