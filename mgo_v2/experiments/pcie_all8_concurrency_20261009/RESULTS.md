# PCIe host-to-device concurrency

Pinned rank-private host source; 256 sequential asynchronous copies per repeat; two unfiltered repeats. Each GPU worker was bound to its own CPU core. This reports effective service under the measured host and GPU conditions, not PCIe theoretical line rate or model E2E speed. Ranges include all combinations of that cardinality and both repeats.

| Payload | Active GPUs | Subsets | Aggregate GiB/s, median [min, max] | Rank speed / solo, median [min, max] |
|---|---:|---:|---:|---:|
| Qwen expert | 1 | 8 | 50.27 [48.56, 50.35] | 1.000 [1.000, 1.000] |
| Qwen expert | 2 | 28 | 100.35 [96.94, 100.62] | 0.999 [0.982, 1.018] |
| Qwen expert | 3 | 56 | 149.19 [129.51, 150.70] | 0.997 [0.858, 1.016] |
| Qwen expert | 4 | 70 | 197.68 [138.24, 199.18] | 0.987 [0.686, 1.013] |
| Qwen expert | 5 | 56 | 224.04 [166.98, 246.33] | 0.978 [0.676, 1.006] |
| Qwen expert | 6 | 28 | 260.20 [203.63, 274.52] | 0.910 [0.681, 0.995] |
| Qwen expert | 7 | 8 | 269.94 [243.17, 300.34] | 0.868 [0.691, 0.965] |
| Qwen expert | 8 | 1 | 276.35 [276.10, 276.60] | 0.770 [0.686, 0.845] |
| DeepSeek expert | 1 | 8 | 50.78 [50.74, 50.82] | 1.000 [1.000, 1.000] |
| DeepSeek expert | 2 | 28 | 101.34 [98.19, 101.61] | 0.998 [0.978, 1.001] |
| DeepSeek expert | 3 | 56 | 151.09 [129.15, 152.23] | 0.997 [0.854, 1.000] |
| DeepSeek expert | 4 | 70 | 199.80 [139.07, 201.76] | 0.988 [0.685, 0.999] |
| DeepSeek expert | 5 | 56 | 226.72 [173.95, 249.10] | 0.976 [0.685, 0.998] |
| DeepSeek expert | 6 | 28 | 263.36 [206.96, 277.18] | 0.910 [0.684, 0.994] |
| DeepSeek expert | 7 | 8 | 272.80 [244.49, 303.72] | 0.861 [0.689, 0.965] |
| DeepSeek expert | 8 | 1 | 278.37 [278.25, 278.49] | 0.763 [0.685, 0.842] |

## Per-rank solo versus 8-way

| GPU | Qwen solo | Qwen 8-way | Qwen ratio | DeepSeek solo | DeepSeek 8-way | DeepSeek ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.40 | 34.61 | 0.687 | 50.86 | 34.84 | 0.685 |
| 1 | 50.42 | 34.58 | 0.686 | 50.82 | 34.83 | 0.685 |
| 2 | 50.38 | 34.70 | 0.689 | 50.84 | 34.99 | 0.688 |
| 3 | 49.53 | 34.62 | 0.699 | 50.86 | 34.90 | 0.686 |
| 4 | 50.36 | 42.48 | 0.843 | 50.84 | 42.69 | 0.840 |
| 5 | 50.43 | 42.44 | 0.842 | 50.84 | 42.59 | 0.838 |
| 6 | 50.42 | 42.60 | 0.845 | 50.85 | 42.76 | 0.841 |
| 7 | 50.35 | 42.57 | 0.845 | 50.83 | 42.82 | 0.842 |

All bandwidth columns use GiB/s.

## Combination effect

| Payload | GPU 0+1+3 aggregate | GPU 0+1+4 aggregate | All 8 aggregate | All 8 / sum of solos |
|---|---:|---:|---:|---:|
| Qwen expert | 131.05 | 148.93 | 276.35 | 68.7% |
| DeepSeek expert | 131.47 | 148.89 | 278.37 | 68.4% |

## Two four-GPU groups and all eight

| Payload | GPUs 0–3 aggregate | GPUs 4–7 aggregate | All 8 aggregate | GPUs 0–3 per-rank speed in all 8 | GPUs 4–7 per-rank speed in all 8 |
|---|---:|---:|---:|---:|---:|
| Qwen expert | 138.27 | 183.50 | 276.35 | 34.62 | 42.52 |
| DeepSeek expert | 139.10 | 185.73 | 278.37 | 34.87 | 42.73 |

The two physical GPU groups have different effective H2D rates under these conditions. This observation does not identify the limiting PCIe, NUMA, IOMMU, or host-memory component.

Maximum observed worker start skew: 0.100 ms.
Individual rank and subset results, including full ranges, are in `RANK_COMBINATIONS.csv` and `RESULTS.json`.
