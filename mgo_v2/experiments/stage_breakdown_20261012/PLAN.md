# Stage-wise cost breakdown (serial default environment), Qwen3 and DeepSeek R8

Environment: R8, ShareGPT local B16, L512/O64, cache 30%, serial per-expert executor (host waits for all
demand H2D before expert compute; H2D overlaps the forward all-to-all), inline one-pass H2D, prefetch OFF.
Qwen: post-generation diagnostic pass (live tokens) after one timed target. DeepSeek: instrumented
teacher-token target (MGO_DEEPSEEK_CACHE_DIAG=1).

Cells (5 arms x 2 models), 4 interleaved repetitions each:

| Arm | Stage 1 quota | Stage 2 placement |
|---|---|---|
| NEAR (`LA_CA_NEAR`) | balanced, remainder to GPUs 0-3 | Near |
| FAST (`NEAR_FAST`) | balanced, remainder to GPUs 4-7 | Near |
| FAST+Random (`FAST_RANDOM`) | FAST | random |
| FAST+Worst (`FAST_WORST`) | FAST | worst (max compute imbalance, then min locality) |
| HAQ_FAST | hit-aware water-fill with per-rank copy cost (cap ceil(m/R)) | Near |

HAQ_FAST changes stage 1 (hit-aware) instead of keeping the FAST rows. Its copy-cost constants
(250/230 us) were calibrated on Qwen 9 MiB experts and are reused unchanged for DeepSeek's 16.5 MiB experts.

Figure panels:
- (a) H2D, NEAR vs FAST: H2D straggler = sum over decode layers of max_r(H2D wait), plus max-mean.
- (b) EP cost for the FAST-quota placements and HAQ_FAST: compute imbalance = sum over layers of
  max_r - mean_r expert compute; comm cost = mean_r(dispatch+return spans) - own imbalance.
