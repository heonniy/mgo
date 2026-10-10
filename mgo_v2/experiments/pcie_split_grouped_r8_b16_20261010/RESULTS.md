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

## Diagnostic follow-up (post-generation instrumented pass, 1 live-token run per policy)

`run_diag.py` (`--post-generation-diagnostic`), analysis in `analyze_diag.py` -> `DIAG_ANALYSIS.json`.
One diagnostic run per policy. Copy events add overhead; tokens are live, not teacher-forced.

1. **In-run DMA asymmetry is ~1.08-1.10x, not 1.25x.** Mean pure DMA per 9 MiB copy (begin->done on the copy stream):
   NEAR 0.254 / 0.235 ms, FAST 0.248 / 0.230 ms, SPLIT 0.255 / 0.231 ms (GPUs 0-3 / 4-7).
   GPUs 4-7 run at ~0.23 ms in the model run versus 0.20 ms in the synchronized burst.
   Per-rank H2D busy time is most even under FAST (5.9-6.4 s). SPLIT pushes 4-7 to 6.2-6.8 s
   versus 5.6-6.1 s on 0-3, so it overshoots.
2. **Copies are submitted one at a time by a Python staging thread.** Median enqueue->submit is about 0.33 ms for
   the first copy of a layer, then grows ~0.17-0.2 ms per copy (≈ DMA pace). Ranks given more copies
   also wait longer in submission (FAST/SPLIT 4-7 mean 1.41/1.49 ms vs 0-3 1.09/1.08 ms). Slot dependencies
   are negligible (0.01-0.04 ms). Copies therefore trickle out rather than burst, which is why the
   synchronized microbenchmark overstates group contention.
3. **The per-layer straggler is the rank with the slowest host submission, and it changes between runs.**
   Share of layers where a rank had the largest (exposed H2D wait + grouped GEMM):
   NEAR r1 41% / r6 23% / r0 23%; FAST r5 38% / r7 21% / r2 16%; SPLIT r5 63% / r4 17% / r6 14%.
   The top straggler has the highest median submit delay in its run (e.g. r5: 1.83 ms FAST, 2.07 ms SPLIT,
   vs ~0.8-1.2 ms elsewhere). Host-side submission jitter is a larger lever than the PCIe group split.
4. Diagnostic wall time agrees with the timed order: NEAR 26.76 s, FAST 26.65 s, SPLIT 27.59 s.

Implication: PCIe-aware quota is second order once FAST balances the groups. Next lever is the copy
submission path: submit a layer's demand copies in one batch from native code instead of
per-copy Python wakeups. That would cut the ~0.3 ms start latency and the rank-specific jitter
for every policy.
