# R8 grouped: NEAR vs FAST vs NEAR_SPLIT

Qwen3-30B ShareGPT R8/C30/local-B16/input512/output64, `new_OURS` grouped
hit-then-miss, prefetch OFF, frozen Near teacher tokens. Three interleaved rounds
with rotated order, each job with 2 unfiltered target repeats (6 samples per policy).
Source commit `2d5ccf1`. NEAR is the baseline.

| Policy | n | TPOT median [range], s | vs NEAR | per-round vs NEAR | E2E vs NEAR |
|---|---:|---:|---:|---|---:|
| NEAR | 6 | 0.2854 [0.2832, 0.2888] | 0 | — | 0 |
| FAST | 6 | 0.2804 [0.2781, 0.2840] | **-1.75%** | -1.16, -2.21, -1.44 | -1.82% |
| SPLIT | 6 | 0.2858 [0.2838, 0.2866] | +0.14% | -0.44, -0.08, +0.47 | +0.27% |

Full table and samples: `RESULTS_TABLE.md`, `RESULTS.json` (`python summarize.py`).

## Why SPLIT did not help

The microbenchmark (all 8 GPUs start copying at the same instant) measured GPUs
4-7 as ~1.25x faster than 0-3, and SPLIT gives 4-7 about 1.25-1.29x the copies.
In the real decode, copies start at different times and overlap hit compute,
so the effective asymmetry is smaller. The per-rank host wait for demand misses
(`grouped_executor_counts.serial_wait_wall_ns`, median over the 6 samples of the
largest rank in each group) shows this:

| Policy | GPU 0-3 max wait, s | GPU 4-7 max wait, s | bottleneck |
|---|---:|---:|---|
| NEAR | 2.60 | 1.79 | 0-3 |
| FAST | 2.33 | 2.08 | 0-3 (slightly) |
| SPLIT | 1.89 | 2.43 | 4-7 |

SPLIT overshoots: GPUs 4-7 become the bottleneck. The balance point lies
between FAST and SPLIT, much closer to FAST. A per-rank wait fit puts it near
25.1k / 27.7k copies per rank, versus FAST's ~25.7k / 27.2k. Equalizing the two
groups' maximum wait would save at most ~0.1-0.15 s of ~18 s decode (<1% TPOT).
Most of the PCIe headroom in this runtime is already captured by FAST.

The SPLIT table also always gave the within-group remainder to GPU 4. As a result,
GPU 4 had the most copies (30.4k vs 28.2k on GPU 7) and the longest wait.

Microbenchmark H2D gains therefore overstate end-to-end gains. A quota table should be calibrated from in-run
wait counters, not isolated synchronized copy bursts.
