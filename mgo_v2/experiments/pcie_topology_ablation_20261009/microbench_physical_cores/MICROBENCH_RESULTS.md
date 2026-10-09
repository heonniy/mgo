# Physical PCIe microbenchmark — 2026-10-09

Completed on physical GPUs 0,1,4,5. One 9,437,184-byte BF16 payload per fetch; one shared pinned source per NUMA node. All local/remote mappings have sampled page-location receipts. Every isolated sample ran before NCCL was initialized. Forty-one cells, five warmups and thirty timed samples per cell; all warmup and timed rows retained.

| Comparison | Rank/skew median (ms) | Group median (ms) | Ratio | Paired 95% CI |
|---|---:|---:|---:|---:|
| six_0 / six_1 | 3.1609 | 2.4177 | 1.3074 | 1.3028–1.3169 |
| six_0 / six_2 | 3.1609 | 2.4098 | 1.3117 | 1.3030–1.3185 |
| scale_2_rank / scale_2_group | 1.6276 | 0.8842 | 1.8408 | 1.8209–1.8535 |
| scale_4_rank / scale_4_group | 1.6511 | 1.6455 | 1.0034 | 0.9972–1.0067 |
| scale_6_rank / scale_6_group | 3.1625 | 2.4199 | 1.3068 | 1.3031–1.3140 |
| scale_8_rank / scale_8_group | 3.1851 | 3.1883 | 0.9990 | 0.9951–1.0015 |
| scale_10_rank / scale_10_group | 4.7022 | 3.9608 | 1.1872 | 1.1860–1.1913 |
| scale_14_rank / scale_14_group | 6.2354 | 5.4955 | 1.1346 | 1.1328–1.1368 |
| scale_50_rank / scale_50_group | 20.0699 | 19.3465 | 1.0374 | 1.0365–1.0381 |
| scale_62_rank / scale_62_group | 24.6890 | 23.9675 | 1.0301 | 1.0294–1.0307 |

The ratio describes isolated copy makespan, not model TPOT. Thirty randomized paired repeat blocks support a 10,000-draw bootstrap interval for the ratio of medians. Per-rank CUDA event service and host completion, group bandwidth, p10/p90/p99 and all raw rows are separate artifacts. Group bandwidth is decimal GB/s from common release to that group’s last active rank.

Directional peer costs use measured two-rank NCCL broadcast service, 9-MiB-minus-4-KiB median transfer slopes, normalized to integer scale 1000. This calibration contains only sender and receiver; it is not a full all-to-all latency model. Across groups NCCL selected SHM; within groups P2P/CUMEM. Those transport differences must remain visible in the topology-weighted policy interpretation.

Attempt 1 failed before timing because CUDA host registration had no current context. Attempt 2 completed samples but failed in PyTorch 2.5 P2P communicator teardown under per-rank single-GPU visibility. Attempt 3 replaced calibration P2P calls with two-rank collectives, completed all measurements, unregistered sources, destroyed groups, and exited cleanly. Earlier attempts remain under the raw output root.

Raw root: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/G2_physical_core_confirmation_attempt1`. No model-generation performance is claimed by this experiment.
