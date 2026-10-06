# R4 expert-fetch path results

Completed on GPUs 0,1,4,5 with three measured repeats after one warmup. 28 condition/burst combinations, two paths, 168 measured synchronized bursts and 288 active-rank samples. All samples retained. Successful attempt took 14.52 seconds including initialization.

First attempt failed before measurement because the runner omitted the existing numba dependency path. Second attempt was stopped by the agent after initialization stalled; its generic owner STOP receipt is an operational cancellation, not a new user instruction. Restoring the previously established NCCL_CUMEM_ENABLE=0 IPC baseline allowed completion. The stall mechanism was not isolated. All failed receipts and raw directories are preserved.

## Critical active-rank wall time

Each repeat takes the maximum over active ranks. Values below are median [min, max] milliseconds. Spread is (max-min)/mean, flagged when >5%; no samples were discarded or extra repeats added.

| Condition | Experts | Current staging ms | Direct pinned ms | Reduction | >5% spread |
|---|---:|---:|---:|---:|---|
| iso0 | 1 | 1.339 [1.329, 1.422] | 0.228 [0.228, 0.229] | 82.95% | current_staging |
| iso0 | 8 | 9.854 [8.919, 10.087] | 1.498 [1.498, 1.526] | 84.79% | current_staging |
| iso0 | 16 | 14.101 [13.558, 15.469] | 2.931 [2.926, 2.944] | 79.22% | current_staging |
| iso0 | 32 | 25.686 [25.363, 25.757] | 5.804 [5.798, 5.832] | 77.40% | none |
| iso1 | 1 | 1.776 [1.756, 1.876] | 0.226 [0.226, 0.238] | 87.28% | current_staging, direct_pinned |
| iso1 | 8 | 11.337 [10.713, 11.728] | 1.494 [1.484, 1.497] | 86.82% | current_staging |
| iso1 | 16 | 15.146 [13.991, 16.868] | 2.940 [2.924, 2.953] | 80.59% | current_staging |
| iso1 | 32 | 24.733 [24.629, 25.629] | 5.830 [5.803, 5.833] | 76.43% | none |
| iso2 | 1 | 2.204 [2.097, 2.285] | 0.234 [0.230, 0.236] | 89.39% | current_staging |
| iso2 | 8 | 13.499 [11.911, 14.120] | 1.566 [1.547, 1.566] | 88.40% | current_staging |
| iso2 | 16 | 18.018 [16.023, 19.571] | 3.078 [3.077, 3.092] | 82.92% | current_staging |
| iso2 | 32 | 27.657 [26.520, 29.461] | 5.961 [5.956, 5.984] | 78.45% | current_staging |
| iso3 | 1 | 1.778 [1.520, 1.828] | 0.230 [0.225, 0.239] | 87.03% | current_staging, direct_pinned |
| iso3 | 8 | 11.399 [9.837, 11.822] | 1.502 [1.500, 1.517] | 86.83% | current_staging |
| iso3 | 16 | 14.882 [14.059, 16.629] | 2.965 [2.965, 2.974] | 80.07% | current_staging |
| iso3 | 32 | 24.290 [24.237, 24.953] | 5.821 [5.820, 5.852] | 76.04% | none |
| pair01 | 1 | 2.003 [1.952, 2.074] | 0.233 [0.226, 0.233] | 88.38% | current_staging |
| pair01 | 8 | 12.709 [11.500, 12.723] | 1.493 [1.491, 1.499] | 88.25% | current_staging |
| pair01 | 16 | 15.989 [14.638, 19.040] | 2.979 [2.976, 3.025] | 81.37% | current_staging |
| pair01 | 32 | 24.683 [23.261, 31.051] | 5.819 [5.817, 5.847] | 76.43% | current_staging |
| pair23 | 1 | 2.163 [2.117, 2.188] | 0.227 [0.225, 0.228] | 89.53% | none |
| pair23 | 8 | 13.352 [12.272, 14.461] | 1.555 [1.542, 1.557] | 88.36% | current_staging |
| pair23 | 16 | 18.144 [16.549, 20.351] | 3.108 [3.095, 3.131] | 82.87% | current_staging |
| pair23 | 32 | 26.683 [25.336, 28.925] | 6.055 [6.036, 6.059] | 77.31% | current_staging |
| all4 | 1 | 1.892 [1.858, 1.935] | 0.339 [0.333, 0.345] | 82.07% | none |
| all4 | 8 | 13.165 [11.829, 13.539] | 1.655 [1.644, 1.662] | 87.43% | current_staging |
| all4 | 16 | 18.235 [17.898, 19.784] | 3.134 [3.131, 3.168] | 82.81% | current_staging |
| all4 | 32 | 35.669 [35.508, 36.161] | 6.074 [6.060, 6.088] | 82.97% | none |

## Per-rank 32-expert comparison

| GPU | Isolated current ms | All4 current ms | Slowdown | Isolated direct ms | All4 direct ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 25.686 | 35.626 | 38.7% | 5.804 | 5.964 |
| 1 | 24.733 | 35.212 | 42.4% | 5.830 | 5.831 |
| 4 | 27.657 | 35.669 | 29.0% | 5.961 | 6.074 |
| 5 | 24.290 | 35.427 | 45.9% | 5.821 | 5.966 |

## Interpretation and limits

All4/32 current-path rank medians cluster at 35.21–35.67 ms: no single rank dominates this burst. Relative to isolated ranks, current-path slowdown is about 29–46%. Direct-pinned all4 critical latency is 6.074 ms versus 35.669 ms current staging (82.97% reduction). The direction supports shared staging/host-path overhead, but does not identify PCIe saturation versus CPU memory bandwidth versus scheduling as a unique cause.

All4/32 per-expert staging medians are 0.952–0.998 ms; current DMA medians 0.211–0.359 ms, versus direct DMA 0.177–0.181 ms. Queue delay and summed stage/DMA durations overlap and must not be added as independent wall-time components.

Direct pinned is an upper-bound microbenchmark with only 288 MiB pinned per rank, not a full 54 GiB expert-store implementation. Direct mode also bypasses the scheduler/queue and uses a different submission loop, so the entire gain cannot be attributed only to the staging memcpy. Current-path profiling is enabled. The scheduler is created before timing for each burst; full-model persistent-worker behavior is not reproduced. No model forward, TTFT, TPOT, output validation or full-store pinned-memory feasibility was measured.

All ranks access the same selected expert keys; real policy-dependent expert sets and sustained multi-layer traffic may differ. Execution PASS verifies completion and trace counts, not copied-payload equality. CPU affinity and source keys are preserved in rank receipts. The 32-expert all4 critical spread is 1.83% current and 0.46% direct; some smaller bursts have >5% spread and are flagged in the table.

The >=10% microbenchmark criterion is met: a registered/pinned source path warrants a separate implementation experiment. This is not evidence of an 83% end-to-end inference gain. No full main-table or R8 run was launched. Owned model-forward load was restored only on GPUs 0,1,4,5 after measurement.
