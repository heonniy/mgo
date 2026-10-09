# Qwen ShareGPT R2/R8 main table

Input 512, 64 greedy output tokens, B16 per GPU, C30 expert budget. TPOT is seconds per decoded token and includes attention; TPS is global output tokens divided by E2E seconds. Each cell has two unfiltered target repeats after warmup; if the pair differs by >2% but ≤5% in TPOT or E2E, exactly one third target is added. The reported center is the mean of two or median of three, followed by the full range. R2 and R8 have different global batches (32 and 128), so throughput is not a same-batch scaling comparison. MoE-Infinity uses EAM eviction priorities with speculative transfer admission disabled on both rank counts after the R8 long-target native wait stall; the expert budget is unchanged.

| Ranks | System | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Repeat quality |
|---:|---|---:|---:|---:|---:|---|
| 2 | Ours | 3.904 [2.257, 3.910] | 0.6541 [0.6505, 0.6566] | 44.893 [43.626, 45.110] | 45.620 [45.400, 46.945] | third repeat; median shown |
| 2 | DeepSpeed-ZeRO-Inference | 5.774 [5.465, 6.084] | 4.1158 [4.1044, 4.1272] | 265.070 [264.660, 265.479] | 7.726 [7.714, 7.738] | ≤2% TPOT/E2E pair range |
| 2 | MoE-Infinity-repaired | 4.730 [4.649, 4.811] | 2.5783 [2.5555, 2.6011] | 167.164 [165.645, 168.682] | 12.252 [12.141, 12.364] | ≤2% TPOT/E2E pair range |
| 2 | llama.cpp-sync | 73.855 [73.801, 73.910] | 0.2938 [0.2937, 0.2939] | 92.367 [92.305, 92.428] | 22.173 [22.158, 22.187] | ≤2% TPOT/E2E pair range |
| 8 | Ours | 2.472 [1.816, 3.127] | 0.3866 [0.3820, 0.3912] | 26.829 [25.882, 27.775] | 305.723 [294.936, 316.510] | unstable two-repeat pair (>5%) |
| 8 | DeepSpeed-ZeRO-Inference | 6.502 [5.918, 7.085] | 4.5345 [4.5240, 4.5450] | 292.175 [290.933, 293.417] | 28.038 [27.919, 28.158] | ≤2% TPOT/E2E pair range |
| 8 | MoE-Infinity-repaired | 8.736 [7.925, 9.546] | 3.6290 [3.6279, 3.6301] | 237.363 [236.486, 238.241] | 34.513 [34.385, 34.641] | ≤2% TPOT/E2E pair range |
| 8 | llama.cpp-sync | 308.743 [308.708, 308.777] | 0.8554 [0.8547, 0.8561] | 362.633 [362.554, 362.711] | 22.590 [22.585, 22.595] | ≤2% TPOT/E2E pair range |

All target repeats are preserved without outlier selection in [RESULTS.json](RESULTS.json). The quality flag uses the relative difference of the first two TPOT and E2E values; TTFT variability is visible separately in its range. Raw paths, source commits, and workload hashes are in the same JSON file.

main_OURS has the lowest E2E in both rank counts. At R2, llama.cpp has the lowest TPOT but its approximately 74-second TTFT makes its E2E longer than main_OURS. At R8, main_OURS has the lowest TPOT as well, while its E2E pair is marked unstable rather than treated as a precise point estimate.
