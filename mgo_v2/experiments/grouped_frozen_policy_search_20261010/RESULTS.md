# Grouped `new_OURS`: frozen-route placement comparison

R4 on GPUs 0/1/4/5; ShareGPT input128, 32 decode forwards; prefetch OFF; native C++ expert execution; grouped hit wave then grouped miss wave. One BR-captured route and teacher input per cell were replayed for all policies. Each displayed TPOT is the mean of **two unfiltered runs** in the order BR/Near/Static/Static/Near/BR. Ranges are the full two-run ranges.

| Transport | Cache | B/rank | Seed | BR TPOT (s) | Near TPOT (s) | Static TPOT (s) | Near gain vs BR |
|---|---:|---:|---|---:|---:|---:|---:|
| nvswitch | C30 | 8 | s20/d7 | 0.3597 [0.3593, 0.3601] | 0.3613 [0.3608, 0.3618] | 0.3610 [0.3590, 0.3629] | -0.45% |
| env2 | C30 | 8 | s20/d7 | 0.3532 [0.3532, 0.3532] | 0.3544 [0.3538, 0.3550] | 0.3586 [0.3582, 0.3590] | -0.33% |
| nvswitch | C30 | 8 | s14/d5 | 0.3569 [0.3569, 0.3570] | 0.3579 [0.3579, 0.3579] | 0.3605 [0.3602, 0.3609] | -0.26% |
| env2 | C30 | 8 | s14/d5 | 0.3401 [0.3397, 0.3405] | 0.3400 [0.3399, 0.3401] | 0.3458 [0.3432, 0.3483] | +0.04% |
| nvswitch | C30 | 16 | s22/d3 | 0.4213 [0.4203, 0.4223] | 0.4231 [0.4228, 0.4234] | 0.4216 [0.4216, 0.4216] | -0.43% |
| env2 | C30 | 16 | s22/d3 | 0.4083 [0.4074, 0.4092] | 0.4095 [0.4085, 0.4106] | 0.4123 [0.4121, 0.4124] | -0.29% |
| nvswitch | C30 | 64 | s20/d5 | 0.4870 [0.4863, 0.4877] | 0.4887 [0.4879, 0.4894] | 0.4864 [0.4855, 0.4873] | -0.34% |
| env2 | C30 | 64 | s20/d5 | 0.4978 [0.4962, 0.4994] | 0.5001 [0.4990, 0.5012] | 0.4957 [0.4939, 0.4974] | -0.46% |
| nvswitch | C60 | 8 | s20/d7 | 0.2877 [0.2872, 0.2882] | 0.2892 [0.2886, 0.2897] | 0.2924 [0.2920, 0.2929] | -0.51% |
| nvswitch | C60 | 8 | s14/d5 | 0.2778 [0.2771, 0.2784] | 0.2781 [0.2776, 0.2787] | 0.2799 [0.2788, 0.2810] | -0.13% |
| nvswitch | C60 | 16 | s22/d3 | 0.3243 [0.3239, 0.3247] | 0.3258 [0.3254, 0.3262] | 0.3294 [0.3288, 0.3301] | -0.46% |
| nvswitch | C60 | 64 | s20/d5 | 0.3889 [0.3866, 0.3913] | 0.3922 [0.3906, 0.3938] | 0.3946 [0.3945, 0.3946] | -0.83% |

The gain column is `(BR − Near) / BR`; a positive value favors Near. A best observed seed is not a global optimum. Repeat ranges and the fresh confirmation determine whether the apparent gain is credible.

Largest screen gain: **ShareGPT_R4_C30_B8_L128_O33_s14_d5**, +0.04%. The two-run ranges do not separate.

## Fresh confirmation of the largest screened mean gain

The same env2 C30/B8 seed 14/5 and identical route/teacher were rerun in a fresh guarded job. BR was 0.2690 [0.2606, 0.2773] and Near was 0.2676 [0.2603, 0.2750] s/token, an observed Near mean gain of +0.50%. Within that job, BR repeats differed by 6.2% and Near by 5.5%; their ranges overlap. Both absolute TPOT means also shifted about 21% below the first job, despite identical route and H2D bytes. Thus this seed is the best observed positive candidate, **not a reliable Near speedup**. The full 12-cell screen found no stable Near winner under these grouped settings.

A later quiet-host rerun, after stopping the managed loads on GPUs 2/3/6/7, and the same-route Ready-First comparison are in [QUIET_PAIR.md](../main_ours_static_placement_20261010/QUIET_PAIR.md). The quiet grouped run favored BR by 0.58%; Ready-First favored Near by 1.90%.

Full per-run values, peer/H2D bytes, decode hit rates, token agreement, and raw receipt paths are in [RESULTS.json](RESULTS.json).
