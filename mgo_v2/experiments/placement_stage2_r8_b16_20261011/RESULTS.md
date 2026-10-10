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

## Why placement barely matters in grouped mode (diagnostic pass, 1 live-token run per arm)

`run_diag.py` -> `analyze_diag.py` -> `DIAG_ANALYSIS.json`. Decode-only sums.

| | FAST+Near | FAST+Random | FAST+Worst |
|---|---:|---:|---:|
| grouped GEMM per layer, median per rank | 0.22 ms | 0.22 ms | 0.22 ms |
| exposed H2D wait (rank mean) | 2.12 s | 2.13 s | 2.24 s |
| return a2a span (rank mean) | 2.45 s | 2.51 s | **3.21 s** |
| route all-gather span (rank mean) | 0.99 s | 1.11 s | **1.47 s** |
| forward bytes (all ranks) | 862 MB | 925 MB | 724 MB |
| return bytes per rank, max/mean | 1.03 | 1.04 | **1.20** |
| diagnostic wall | 24.74 s | 25.01 s | 25.59 s |

1. **Grouped GEMM is launch-bound.** Every rank's per-layer grouped GEMM is ~0.22 ms in all three
   arms, even though the critical/mean expert rows differ 1.14 / 1.53 / 2.82. Per-rank totals show
   occasional +0.35 s spikes on 2-3 random ranks in every arm. Those are noise, not placement. So
   the compute-balance objective (Near's LA term) has nothing to optimize here.
   In serial mode per-expert compute does scale with rows, which is why Near wins there.
2. **Placement shows up only through communication hotspots.** Worst sends the fewest bytes,
   because concentrating hot experts reduces distinct destination ranks per token. But it makes
   one rank return far more (return max/mean 1.20), and every other rank waits for it:
   return a2a +0.76 s and all-gather +0.48 s versus Near.
3. **Near vs Random.** Near sends ~7% fewer bytes, and in this diagnostic its forward a2a completes
   0.37 s sooner. Neither creates a hotspot, and the timed TPOT gap is within session noise.
   Near's locality term (rank-local rows) is only a proxy. What a2a time follows is the most-loaded
   sender/receiver per layer.

So in grouped mode stage 2 only has to avoid communication hotspots. A placement objective
would need to target per-layer max a2a volume, not row balance, to matter here.
