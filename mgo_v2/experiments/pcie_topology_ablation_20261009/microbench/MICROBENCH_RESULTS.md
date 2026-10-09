# Physical PCIe microbenchmark — 2026-10-09

Completed on physical GPUs 0,1,4,5. One 9,437,184-byte BF16 payload per fetch; one shared pinned source per NUMA node. All local/remote mappings have sampled page-location receipts. Every isolated sample ran before NCCL was initialized. Forty-one cells, five warmups and thirty timed samples per cell; all warmup and timed rows retained.

| Comparison | Rank/skew median (ms) | Group median (ms) | Ratio | Paired 95% CI |
|---|---:|---:|---:|---:|
| six_0 / six_1 | 3.1690 | 2.4299 | 1.3042 | 1.2991–1.3075 |
| six_0 / six_2 | 3.1690 | 2.4339 | 1.3020 | 1.2976–1.3066 |
| scale_2_rank / scale_2_group | 1.6345 | 0.9100 | 1.7962 | 1.7680–1.8168 |
| scale_4_rank / scale_4_group | 1.6746 | 1.6714 | 1.0019 | 0.9951–1.0080 |
| scale_6_rank / scale_6_group | 3.1739 | 2.4331 | 1.3045 | 1.2972–1.3102 |
| scale_8_rank / scale_8_group | 3.2104 | 3.2138 | 0.9990 | 0.9949–1.0040 |
| scale_10_rank / scale_10_group | 4.7001 | 3.9775 | 1.1817 | 1.1784–1.1845 |
| scale_14_rank / scale_14_group | 6.2483 | 5.5130 | 1.1334 | 1.1307–1.1369 |
| scale_50_rank / scale_50_group | 20.0810 | 19.3732 | 1.0365 | 1.0359–1.0374 |
| scale_62_rank / scale_62_group | 24.6902 | 24.0007 | 1.0287 | 1.0275–1.0296 |

The ratio describes isolated copy makespan, not model TPOT. Thirty randomized paired repeat blocks support a 10,000-draw bootstrap interval for the ratio of medians. Per-rank CUDA event service and host completion, group bandwidth, p10/p90/p99 and all raw rows are separate artifacts. Group bandwidth is decimal GB/s from common release to that group’s last active rank.

Directional peer costs use measured two-rank NCCL broadcast service, 9-MiB-minus-4-KiB median transfer slopes, normalized to integer scale 1000. This calibration contains only sender and receiver; it is not a full all-to-all latency model. Across groups NCCL selected SHM; within groups P2P/CUMEM. Those transport differences must remain visible in the topology-weighted policy interpretation.

Attempt 1 failed before timing because CUDA host registration had no current context. Attempt 2 completed samples but failed in PyTorch 2.5 P2P communicator teardown under per-rank single-GPU visibility. Attempt 3 replaced calibration P2P calls with two-rank collectives, completed all measurements, unregistered sources, destroyed groups, and exited cleanly. Earlier attempts remain under the raw output root.

Raw root: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/G2_microbench_attempt3`. No model-generation performance is claimed by this experiment.
