# R8 serial GEMM + inline H2D + FAST quota: Near vs worst placement

Three rounds with alternating order, 6 samples per arm. Full table: `RESULTS_TABLE.md`.

| Arm | TPOT median [range], s | vs FAST+Near | per round | max H2D wait, GPUs 0-3 / 4-7 |
|---|---:|---:|---|---|
| FAST+Near | 0.4247 [0.4212, 0.4288] | 0 | — | 4.16 / 4.00 s |
| FAST+Worst | 0.5170 [0.5077, 0.5221] | **+21.75%** | +21.33, +21.38, +21.92 | 3.95 / 3.86 s |

- In serial mode stage-2 placement is decisive: the worst placement is +21.8% slower, consistently
  in every round. The ranges are far apart. In grouped mode the same placement cost only ~1-2%.
- The loss is compute, not H2D. Quota and fetch counts are identical (211,346 vs 211,341), and the
  worst arm's max H2D wait is even slightly lower. Worst puts the heaviest misses (most tokens) on one
  rank (CPU proxy: critical/mean expert rows 2.82 vs 1.14). Serial per-expert execution makes every
  other rank wait for it in each layer.
- Same repeat-position effect as in `../serial_inline_r8_b16_20261011`: the first target is ~1.5-2% slower.
