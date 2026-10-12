# Stage-2 placement cost: load imbalance vs communication (serial + inline H2D, FAST quota)

One diagnostic job per arm and model (`run_diag.py` -> `analyze.py` -> `DECOMPOSITION.json`).
Decode-only sums over layers, in seconds. own = H2D wait + expert compute per rank and layer. Ranks that
finish early wait inside the next collective, so pure comm = mean_r(comm) - (max_r(own) - mean_r(own)).

| Model | Arm | mean compute | compute imbalance | mean H2D wait | own imbalance | pure comm | bytes (decode, all ranks) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3 | FAST+Near | 8.14 | 1.30 | 3.13 | 1.46 | 8.01 | 14.1 GB |
| Qwen3 | FAST+Random | 8.19 | 2.24 | 3.27 | 2.37 | 8.53 | 14.7 GB |
| Qwen3 | FAST+Worst | 8.20 | **6.59** | 3.13 | **6.71** | **10.99** | **11.0 GB** |
| DeepSeek | FAST+Near | 2.84 | 0.70 | 2.63 | 0.82 | 3.96 | 6.6 GB |
| DeepSeek | FAST+Random | 2.85 | 0.92 | 2.58 | 1.02 | 4.09 | 6.8 GB |
| DeepSeek | FAST+Worst | 2.82 | **1.74** | 2.63 | **1.79** | **4.37** | **6.3 GB** |

Change versus FAST+Near:

| Model | Arm | Δ load imbalance | Δ pure comm | Δ mean compute / wait | sum | measured (timed runs) |
|---|---|---:|---:|---:|---:|---:|
| Qwen3 | Random | +0.91 s | +0.52 s | +0.06 / +0.14 | +1.63 s | +0.6 s (+2.2% TPOT) |
| Qwen3 | Worst | **+5.25 s** | **+2.98 s** | +0.07 / 0.00 | +8.30 s | +5.8 s (+21.8%) |
| DeepSeek | Random | +0.20 s | +0.13 s | +0.01 / -0.05 | +0.28 s | +0.2 s (+1.1%) |
| DeepSeek | Worst | **+0.97 s** | **+0.40 s** | -0.02 / 0.00 | +1.35 s | +1.4 s (+8.5%) |

- Total compute and H2D work are unchanged by placement (mean compute within 1%, mean wait within
  0.15 s). Placement only moves where that work lands.
- **About two-thirds of the placement cost is load imbalance**, and nearly all of it is expert compute
  imbalance. About one-third is communication.
- **The comm cost is not volume.** The worst placement sends the fewest bytes (Qwen -22%, DeepSeek -5%),
  but its communication still takes longer. Concentrating hot experts on one rank shrinks the number of
  distinct destinations but creates a receive/return hotspot, and the collective waits for that rank.
- The worst-placement cost is larger for Qwen because its per-layer expert compute is ~3x DeepSeek's
  (8.1 s vs 2.8 s of decode). The same concentration therefore produces a ~5x larger imbalance in seconds.
- Caveat: one instrumented run per arm, and diagnostic events add overhead. The span-based sum
  overshoots the timed difference for Qwen Random/Worst (+1.6 vs +0.6 s, +8.3 vs +5.8 s). Treat the
  imbalance/comm split as the result, not the absolute seconds.
- Side observation: DeepSeek spends ~113 ms per decode step on host route/controller work before the dispatch
  (`route_to_dispatch_host_ms`), out of a ~317 ms instrumented step.
