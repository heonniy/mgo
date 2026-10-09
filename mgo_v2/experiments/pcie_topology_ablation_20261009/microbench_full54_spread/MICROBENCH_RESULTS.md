# Physical PCIe microbenchmark — 2026-10-09

Pinned source per NUMA node: 54 GiB. Address layout: spread. Unique physical source: 108 GiB.

Completed on physical GPUs 0,1,4,5. One 9,437,184-byte BF16 payload per fetch; one shared pinned source per NUMA node. All local/remote mappings have sampled page-location receipts. Every isolated sample ran before NCCL was initialized. Forty-one cells, five warmups and thirty timed samples per cell; all warmup and timed rows retained.

| Comparison | Rank/skew median (ms) | Group median (ms) | Ratio | Paired 95% CI |
|---|---:|---:|---:|---:|
| six_0 / six_1 | 3.1682 | 2.4137 | 1.3126 | 1.3075–1.3176 |
| six_0 / six_2 | 3.1682 | 2.4171 | 1.3108 | 1.3052–1.3147 |
| scale_2_rank / scale_2_group | 1.6313 | 0.8799 | 1.8539 | 1.8240–1.8753 |
| scale_4_rank / scale_4_group | 1.6475 | 1.6442 | 1.0020 | 0.9922–1.0086 |
| scale_6_rank / scale_6_group | 3.1695 | 2.4150 | 1.3124 | 1.3088–1.3170 |
| scale_8_rank / scale_8_group | 3.1852 | 3.1786 | 1.0021 | 0.9982–1.0054 |
| scale_10_rank / scale_10_group | 4.7109 | 3.9522 | 1.1920 | 1.1890–1.1946 |
| scale_14_rank / scale_14_group | 6.2444 | 5.4878 | 1.1379 | 1.1361–1.1396 |
| scale_50_rank / scale_50_group | 20.0821 | 19.3430 | 1.0382 | 1.0370–1.0391 |
| scale_62_rank / scale_62_group | 24.7060 | 23.9712 | 1.0307 | 1.0297–1.0314 |

The ratio describes isolated copy makespan, not model TPOT. Thirty randomized paired repeat blocks support a 10,000-draw bootstrap interval for the ratio of medians. Per-rank CUDA event service and host completion, group bandwidth, p10/p90/p99 and all raw rows are separate artifacts. Group bandwidth is decimal GB/s from common release to that group’s last active rank.

Directional peer costs use measured two-rank NCCL broadcast service, 9-MiB-minus-4-KiB median transfer slopes, normalized to integer scale 1000. This calibration contains only sender and receiver; it is not a full all-to-all latency model. Across groups NCCL selected SHM; within groups P2P/CUMEM. Those transport differences must remain visible in the topology-weighted policy interpretation.

Earlier G2_microbench_attempt1 failed before timing because CUDA host registration had no current context. G2_microbench_attempt2 completed samples but failed in PyTorch 2.5 P2P communicator teardown under per-rank single-GPU visibility. G2_microbench_attempt3 replaced calibration P2P calls with two-rank collectives, completed all measurements, unregistered sources, destroyed groups, and exited cleanly. Those earlier attempts remain under the raw output root; they are not failures of this confirmation job.

Raw root: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/G2_full54_spread_attempt1`. No model-generation performance is claimed by this experiment.
