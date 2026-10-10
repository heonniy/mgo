# R8 grouped (inline H2D): NEAR vs FAST vs HAQ vs HAQ_FAST

All arms use `--inline-demand-h2d`. Grouped hit-then-miss, prefetch OFF, frozen Near
teacher tokens. Two rounds with rotated order, 4 samples per arm. Baseline is NEAR,
which is PCIe-topology-unaware and gives the remainder to the slow GPUs 0-3.
Full table: `RESULTS_TABLE.md`.

| Arm | TPOT median [range], s | vs NEAR | per round | copies per rank, GPUs 0-3 / 4-7 |
|---|---:|---:|---|---|
| NEAR | 0.2804 [0.2792, 0.2824] | 0 | — | 26.6-27.7k / 25.1-26.2k |
| FAST | 0.2745 [0.2736, 0.2749] | **-2.12%** | -1.81, -2.63 | 25.1-26.2k / 26.6-27.8k |
| HAQ | 0.2761 [0.2758, 0.2767] | -1.54% | -1.26, -1.89 | 26.4-26.6k / 26.1-26.4k |
| HAQ_FAST | 0.2745 [0.2742, 0.2758] | **-2.11%** | -1.84, -2.32 | 25.6-25.8k / 27.1-27.2k |

- Adding the per-rank PCIe copy cost improves HAQ from -1.54% to -2.11%, which equals FAST.
- HAQ's hit-awareness adds nothing on top of FAST in grouped mode, because hit compute overlaps H2D.
  HAQ alone beats NEAR only because its hit-driven remainder spreads extras across both groups.
- HAQ_FAST evens copies inside each group (spread under 1%, vs FAST's ~4%). That gives no measurable
  TPOT gain here.
- Combined with the inline result (`../inline_h2d_r8_b16_20261011`), the best configuration so far is
  inline submission with FAST or HAQ_FAST, about -4.7% TPOT versus the original NEAR with staging-thread
  submission.
