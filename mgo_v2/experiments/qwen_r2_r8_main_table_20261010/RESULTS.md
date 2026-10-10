# Qwen ShareGPT R2/R8 main table

Input 512, 64 greedy output tokens, B16 per GPU, C30 expert budget. TPOT is seconds per decoded token and includes attention; TPS is global output tokens divided by E2E seconds. The original R2/R8 extension gives each cell two unfiltered target repeats after warmup; if the pair differs by >2% in TPOT or E2E, exactly one third target is added. The reported center is the mean of two or median of three, followed by the full range. An initial gap above 5% remains flagged even after the third repeat. The R4 rows below are imported from an **earlier, separately measured** C30 sweep, with their own two- or three-repeat protocol. B16 per GPU means global batches of 32, 64 and 128 at R2/R4/R8, so throughput is not a same-batch scaling comparison. The R2/R8 MoE-Infinity rows use EAM eviction priorities with speculative transfer admission disabled after the R8 long-target native wait stall; the expert budget is unchanged. Do not assume the earlier R4 MoE-Infinity run used that later repair setting.

| Ranks | System | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Repeat quality |
|---:|---|---:|---:|---:|---:|---|
| 2 | Ours | 3.904 [2.257, 3.910] | 0.6541 [0.6505, 0.6566] | 44.893 [43.626, 45.110] | 45.620 [45.400, 46.945] | third repeat; median shown |
| 2 | DeepSpeed-ZeRO-Inference | 5.774 [5.465, 6.084] | 4.1158 [4.1044, 4.1272] | 265.070 [264.660, 265.479] | 7.726 [7.714, 7.738] | ≤2% TPOT/E2E pair range |
| 2 | MoE-Infinity-repaired | 4.730 [4.649, 4.811] | 2.5783 [2.5555, 2.6011] | 167.164 [165.645, 168.682] | 12.252 [12.141, 12.364] | ≤2% TPOT/E2E pair range |
| 2 | llama.cpp-sync | 73.855 [73.801, 73.910] | 0.2938 [0.2937, 0.2939] | 92.367 [92.305, 92.428] | 22.173 [22.158, 22.187] | ≤2% TPOT/E2E pair range |
| 4 | Ours | 2.700 [1.767, 3.632] | 0.4948 [0.4935, 0.4960] | 33.870 [33.017, 34.723] | 121.010 [117.962, 124.057] | earlier R4 sweep; 2 repeats |
| 4 | DeepSpeed-ZeRO-Inference | 5.571 [5.451, 6.312] | 4.1219 [4.0909, 4.1347] | 265.995 [263.176, 266.055] | 15.399 [15.395, 15.564] | earlier R4 sweep; 3 repeats |
| 4 | MoE-Infinity-repaired | 5.192 [5.182, 5.219] | 3.2272 [3.2133, 3.2639] | 208.505 [207.659, 210.809] | 19.645 [19.430, 19.725] | earlier R4 sweep; 3 repeats |
| 4 | llama.cpp-sync | 145.070 [145.007, 145.185] | 0.4806 [0.4803, 0.4810] | 175.331 [175.287, 175.486] | 23.361 [23.341, 23.367] | earlier R4 sweep; 3 repeats |
| 8 | Ours | 3.127 [1.816, 3.181] | 0.3912 [0.3820, 0.3923] | 27.775 [25.882, 27.896] | 294.936 [293.659, 316.510] | initial pair >5%; third repeat, median shown |
| 8 | DeepSpeed-ZeRO-Inference | 6.502 [5.918, 7.085] | 4.5345 [4.5240, 4.5450] | 292.175 [290.933, 293.417] | 28.038 [27.919, 28.158] | ≤2% TPOT/E2E pair range |
| 8 | MoE-Infinity-repaired | 8.736 [7.925, 9.546] | 3.6290 [3.6279, 3.6301] | 237.363 [236.486, 238.241] | 34.513 [34.385, 34.641] | ≤2% TPOT/E2E pair range |
| 8 | llama.cpp-sync | 308.743 [308.708, 308.777] | 0.8554 [0.8547, 0.8561] | 362.633 [362.554, 362.711] | 22.590 [22.585, 22.595] | ≤2% TPOT/E2E pair range |

The separately requested fourth R2 main_OURS run is a confirmation, not part of the original three-run primary median: TTFT 3.939 s, TPOT 0.6598 s/token, E2E 45.507 s, TPS 45.004. Its raw receipt and exact comparison with the primary median are in [RESULTS.json](RESULTS.json).

The added R4 target is the **same first 64 tokenized ShareGPT requests** as the R8 target, assigned to four ranks on GPUs 0/1/4/5; R2 uses the first 32. The R4 `main_OURS` row is the later native Ready-First, compiled-index, pinned-source, Near, prefetch-OFF timing recheck in [FULL_CACHE_COMPARISON.md](../qwen_cache_ablation_20261009/FULL_CACHE_COMPARISON.md), **not** the older R4 expanded-table OURS value of 0.7552 s/token. The old run used commit `ac88798` and only `--prefill-optimized --prefill-layout-fast`; the newer run used `de853d9` with native expert/prefill, prefetch OFF and fast decode layout. **None of the 64 aligned requests** had an identical 512-token input sequence. Thus the 0.7552→0.4948 difference combines code and input changes and is not a controlled optimization gain. A separate, matched-input [native executor A/B](../native_fullpath_20261008/RESULTS.md) did show 25.1–27.9% lower TPOT across its three R4 cells, supporting native execution as a real contributor, but it did not test this exact B16/L512 pair. R4 baseline values are the validated C30 rows in [BASELINE_SWEEP_RESULTS.json](../qwen_cache_ablation_20261009/BASELINE_SWEEP_RESULTS.json). Their raw samples, source commits, workload hash and recalculated global TPS are preserved in [R4_ADDITION.json](R4_ADDITION.json). Because R4 was measured on different dates and source commits, the combined table is descriptive, not a paired three-rank-count scaling experiment.

R8 main_OURS had one faster middle run. Its first and third runs differed by 0.27% in TPOT and 0.43% in E2E; the three-run median is close to those two, while the full range remains shown above.

All target repeats are preserved without outlier selection in [RESULTS.json](RESULTS.json). The quality flag uses the relative difference of the first two TPOT and E2E values; TTFT variability is visible separately in its range. Raw paths, source commits, and workload hashes are in the same JSON file.

main_OURS has the lowest E2E in all three rank counts. At R2 and R4, llama.cpp has the lowest TPOT but its long TTFT makes its E2E longer than main_OURS. At R8, main_OURS has the lowest TPOT as well; its initial E2E pair exceeded 5%, so the three-repeat median must retain its full range.
