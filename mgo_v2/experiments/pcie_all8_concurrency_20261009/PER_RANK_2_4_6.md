# Per-GPU effective H2D for 2, 4, and 6 simultaneous GPUs

Every one of the 28 two-GPU, 70 four-GPU, and 28 six-GPU subsets is included at both expert payload sizes in `PER_RANK_2_4_6.xlsx` and `PER_RANK_2_4_6.csv`. Blank GPU cells mean that GPU was inactive.

The rates below are medians of CUDA-event rank-local GiB/s across the subsets in each row. Percentages are median losses against that same physical GPU measured alone at the same payload size. Each subset has two unfiltered repeats.

| Simultaneous GPUs | From GPUs 0–3 | Subsets | Qwen 0–3 GiB/s (loss) | Qwen 4–7 GiB/s (loss) | DeepSeek 0–3 GiB/s (loss) | DeepSeek 4–7 GiB/s (loss) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 6 | — | 50.30 (0.2%) | — | 50.75 (0.2%) |
| 2 | 1 | 16 | 50.34 (0.1%) | 50.39 (0.0%) | 50.80 (0.1%) | 50.83 (0.0%) |
| 2 | 2 | 6 | 49.95 (0.9%) | — | 50.48 (0.7%) | — |
| 4 | 0 | 1 | — | 46.13 (8.5%) | — | 46.51 (8.5%) |
| 4 | 1 | 16 | 50.13 (0.5%) | 49.58 (1.6%) | 50.65 (0.4%) | 50.01 (1.6%) |
| 4 | 2 | 36 | 49.71 (1.4%) | 50.25 (0.3%) | 50.25 (1.2%) | 50.70 (0.3%) |
| 4 | 3 | 16 | 43.65 (13.3%) | 50.32 (0.2%) | 43.91 (13.6%) | 50.75 (0.2%) |
| 4 | 4 | 1 | 34.63 (31.2%) | — | 34.84 (31.5%) | — |
| 6 | 2 | 6 | 48.36 (4.0%) | 45.87 (9.0%) | 48.81 (4.0%) | 46.28 (9.0%) |
| 6 | 3 | 16 | 43.52 (13.6%) | 48.95 (2.8%) | 44.04 (13.4%) | 49.41 (2.8%) |
| 6 | 4 | 6 | 34.79 (30.9%) | 49.98 (0.8%) | 35.00 (31.2%) | 50.47 (0.7%) |

Loss is 100 × (1 − subset speed / solo speed); negative values in the full table are small measured improvements and are preserved.
This is repeated pinned-host H2D copy service, not model inference speed or an isolated PCIe component measurement.
