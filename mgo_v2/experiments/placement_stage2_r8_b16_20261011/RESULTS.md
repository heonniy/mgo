# Stage-2 placement under the FAST quota (R8 grouped, inline H2D)

Two rounds with rotated order, 4 samples per arm. Baseline NEAR = original quota + Near placement.
Full table: `RESULTS_TABLE.md`.

| Arm | TPOT median [range], s | vs NEAR | per round |
|---|---:|---:|---|
| NEAR | 0.2780 [0.2767, 0.2804] | 0 | — |
| FAST + Near | 0.2750 [0.2719, 0.2761] | -1.08% | -1.28, -1.44 |
| FAST + Random | 0.2729 [0.2724, 0.2742] | -1.80% | -2.16, -1.53 |
| FAST + Worst | 0.2783 [0.2776, 0.2809] | +0.13% | -0.03, +0.41 |

CPU check of the same controls (`cpu_check.py`), critical/mean expert rows and remote row fraction:
Near 1.14 / 0.81, Random 1.53 / 0.87, Worst 2.82 / 0.90.

- **A bad placement costs about 1-2%.** FAST+Worst gives back the whole FAST gain. Its range
  lies entirely above FAST+Random (min 0.2776 vs max 0.2742).
- **Near is not better than random placement here.** The FAST+Near and FAST+Random ranges overlap,
  and Random's median is 0.7% lower. FAST+Near itself varies by about 1% between sessions:
  -2.12% in `../haq_fast_r8_b16_20261011`, -1.08% here. So the gap is within run-to-run variance.
  Near's better load balance (1.14 vs 1.53) and locality (81% vs 87% remote) do not turn into TPOT in
  grouped mode. Grouped GEMM is short and overlaps H2D, and the all-to-all cost depends little
  on the remote fraction once most rows are remote anyway.
- Fetch counts are effectively unchanged (211,202-211,346).
- Conclusion: in this regime, stage 1 (FAST: extras to fast GPUs) and the inline H2D submission
  carry the gains. Stage 2 only has to avoid pathological placements.
