# PCIe host-to-device concurrency

Pinned rank-private host source; 256 sequential asynchronous copies per repeat; two unfiltered repeats. Each GPU worker was bound to its own CPU core. This reports effective service under the measured host and GPU conditions, not PCIe theoretical line rate or model E2E speed. Ranges include all combinations of that cardinality and both repeats.

| Payload | Active GPUs | Subsets | Aggregate GiB/s, median [min, max] | Rank speed / solo, median [min, max] |
|---|---:|---:|---:|---:|
| Qwen expert | 1 | 7 | 50.26 [50.13, 50.39] | 1.000 [1.000, 1.000] |
| Qwen expert | 2 | 21 | 100.47 [99.48, 100.68] | 1.000 [0.991, 1.002] |
| Qwen expert | 3 | 35 | 149.60 [128.69, 150.79] | 0.998 [0.854, 1.001] |
| Qwen expert | 4 | 35 | 197.41 [171.93, 199.98] | 0.987 [0.855, 1.001] |
| Qwen expert | 5 | 21 | 240.83 [206.01, 246.08] | 0.976 [0.837, 0.997] |
| Qwen expert | 6 | 7 | 258.33 [255.05, 269.52] | 0.897 [0.849, 0.971] |
| Qwen expert | 7 | 1 | 297.68 [297.31, 298.06] | 0.864 [0.844, 0.870] |
| DeepSeek expert | 1 | 7 | 50.79 [49.99, 50.86] | 1.000 [1.000, 1.000] |
| DeepSeek expert | 2 | 21 | 101.38 [98.08, 101.67] | 0.999 [0.977, 1.008] |
| DeepSeek expert | 3 | 35 | 150.83 [128.80, 152.25] | 0.997 [0.845, 1.007] |
| DeepSeek expert | 4 | 35 | 199.12 [170.73, 201.38] | 0.987 [0.843, 1.005] |
| DeepSeek expert | 5 | 21 | 245.16 [214.26, 248.58] | 0.975 [0.845, 0.999] |
| DeepSeek expert | 6 | 7 | 257.60 [250.71, 274.67] | 0.904 [0.834, 0.971] |
| DeepSeek expert | 7 | 1 | 292.00 [289.53, 294.48] | 0.853 [0.821, 0.857] |

## Per-rank solo versus seven-way

| GPU | Qwen solo | Qwen seven-way | Qwen ratio | DeepSeek solo | DeepSeek seven-way | DeepSeek ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 50.39 | 42.70 | 0.847 | 50.88 | 41.88 | 0.823 |
| 1 | 50.43 | 42.59 | 0.844 | 50.84 | 41.74 | 0.821 |
| 3 | 50.35 | 42.75 | 0.849 | 50.46 | 42.05 | 0.833 |
| 4 | 50.33 | 43.51 | 0.864 | 50.75 | 43.41 | 0.855 |
| 5 | 50.36 | 43.56 | 0.865 | 50.81 | 43.33 | 0.853 |
| 6 | 50.48 | 43.80 | 0.868 | 50.90 | 43.57 | 0.856 |
| 7 | 50.39 | 43.84 | 0.870 | 50.86 | 43.59 | 0.857 |

All bandwidth columns use GiB/s.

## Combination effect

| Payload | GPU 0+1+3 aggregate | GPU 0+1+4 aggregate | All 7 aggregate | All 7 / sum of solos |
|---|---:|---:|---:|---:|
| Qwen expert | 128.80 | 149.23 | 297.68 | 84.4% |
| DeepSeek expert | 128.80 | 149.99 | 292.00 | 82.1% |

Maximum observed worker start skew: 0.109 ms.
Individual rank and subset results, including full ranges, are in `RANK_COMBINATIONS.csv` and `RESULTS.json`.
The 0+1+3 penalty reproduces at both payload sizes and both repeats. This benchmark cannot isolate whether PCIe links, host memory, IOMMU or the virtualized upstream fabric caused it.
