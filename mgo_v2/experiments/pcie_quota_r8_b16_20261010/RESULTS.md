# R8/B16 PCIe quota experiment

Qwen3-30B, ShareGPT, per-GPU B16, input512, output64, C30; native Ready-First `main_OURS`, compiled metadata/layout, prefetch OFF. All policies use the same frozen requests and balanced miss-count quota. The lookup changes only which ranks receive the extra misses. Every primary target starts with an empty expert cache.

| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Token parity |
|---|---:|---:|---:|---:|---:|---|
| Original Near | 3 | 3.102 [1.822, 3.122] | 0.3898 [0.3816, 0.3901] | 27.659 [25.866, 27.697] | 296.177 [295.776, 316.713] | exact |
| Fast-rank quota | 3 | 3.089 [1.826, 3.111] | 0.3868 [0.3809, 0.3906] | 27.478 [25.822, 27.697] | 298.128 [295.772, 317.252] | DIFF |
| PCIe lookup quota | 3 | 3.141 [1.839, 3.150] | 0.3876 [0.3801, 0.3887] | 27.566 [25.788, 27.628] | 297.177 [296.512, 317.672] | DIFF |

Values are the mean of two or median of three unfiltered target repeats; brackets contain the full range. A third was added only when the first-pair TPOT or E2E difference exceeded 2%. The initial-pair gap and all raw receipt paths are in [RESULTS.json](RESULTS.json).

The PCIe and simple fast-rank lookup rows differ for 31 of 129 miss counts. At 12 misses both select `[1, 1, 1, 1, 2, 2, 2, 2]`. This is a measured H2D-service model, not a guarantee of end-to-end gain. The microbenchmark uses pinned 9 MiB copies with no model, NCCL or expert computation. Raw calibration records are under `/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010/`.

These live-generation measurements are exploratory: different placements changed generated tokens, so subsequent decode inputs were not matched. In addition, the optional teacher-token control was added between the first original-Near/fast-rank jobs and later jobs. The control used one clean committed source tree throughout; its comparison is in [FROZEN_RESULTS.md](FROZEN_RESULTS.md).
