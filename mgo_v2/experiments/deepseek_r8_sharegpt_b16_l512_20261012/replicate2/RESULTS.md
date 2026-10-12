# DeepSeek R8 serial + inline H2D: replicate run with FAST+Worst

Fresh-session replicate of `../RESULTS.md`, with FAST+Worst added. Same teacher tokens and setup.
Three rounds with rotated order, 6 samples per arm. Full table: `RESULTS_TABLE.md` (`python summarize_rep2.py`).

| Arm | Stage 1 quota | Stage 2 placement | TPOT median [range], s | vs NEAR | per round |
|---|---|---|---:|---:|---|
| NEAR | balanced, remainder to GPUs 0-3 | Near | 0.2579 [0.2541, 0.2595] | 0 | — |
| FAST | balanced, remainder to GPUs 4-7 | Near | 0.2538 [0.2511, 0.2577] | -1.58% | -0.08, -1.76, -1.52 |
| FAST+Random | FAST | random | 0.2568 [0.2535, 0.2612] | -0.43% | +0.57, -0.07, -0.63 |
| FAST+Worst | FAST | worst (max compute imbalance, then min locality) | 0.2753 [0.2713, 0.2796] | **+6.74%** | +6.33, +7.68, +7.07 |
| RANDOM_QUOTA | unbalanced random | Near | 0.2953 [0.2919, 0.2996] | **+14.49%** | +13.79, +14.90, +16.31 |

## Pooled with the first run (pairs matched by round and repeat position)

| Comparison | pairs | median | pairs with the sign shown |
|---|---:|---:|---:|
| FAST vs NEAR | 12 | **-1.57%** | 11/12 faster |
| FAST+Random vs FAST+Near | 12 | **+1.08%** | 11/12 slower |
| FAST+Worst vs FAST+Near | 6 | **+8.46%** | 6/6 slower |
| RANDOM_QUOTA vs NEAR | 12 | **+14.73%** | 12/12 slower |

- FAST is a small but consistent gain on DeepSeek in serial mode: about -1.6% across two sessions.
  It was -2.2% in the first run and -1.6% here, and one round of each run was close to 0.
- Stage 2 matters. Near beats random by ~1.1%, and the worst placement costs ~8.5% versus Near.
  The worst-placement cost is much smaller than Qwen's (+21.8%). The reason is not isolated (see note).
- Quota balance dominates. The unbalanced random quota is +14.7% in both runs.

Note: DeepSeek-V2-Lite routed experts have moe_intermediate_size 1408 and Qwen3-30B-A3B has 768. A DeepSeek
expert therefore has more FLOPs, but it is routed with top-6 over 64 experts versus top-8 over 128.
Which property sets the smaller worst-placement cost is not isolated here.
