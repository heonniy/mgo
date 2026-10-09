# Physical PCIe microbenchmark — 2026-10-09

Pinned source per NUMA node: 54 GiB. Address layout: compact. Unique physical source: 108 GiB.

Completed on physical GPUs 0,1,4,5. One 9,437,184-byte BF16 payload per fetch; one shared pinned source per NUMA node. All local/remote mappings have sampled page-location receipts. Every isolated sample ran before NCCL was initialized. Forty-one cells, five warmups and thirty timed samples per cell; all warmup and timed rows retained.

| Comparison | Rank/skew median (ms) | Group median (ms) | Ratio | Paired 95% CI |
|---|---:|---:|---:|---:|
| six_0 / six_1 | 3.1671 | 2.4066 | 1.3160 | 1.3106–1.3192 |
| six_0 / six_2 | 3.1671 | 2.4105 | 1.3139 | 1.3089–1.3171 |
| scale_2_rank / scale_2_group | 1.6255 | 0.8790 | 1.8492 | 1.8300–1.8756 |
| scale_4_rank / scale_4_group | 1.6450 | 1.6502 | 0.9968 | 0.9895–1.0059 |
| scale_6_rank / scale_6_group | 3.1618 | 2.4122 | 1.3108 | 1.3053–1.3150 |
| scale_8_rank / scale_8_group | 3.1861 | 3.1775 | 1.0027 | 0.9997–1.0068 |
| scale_10_rank / scale_10_group | 4.7060 | 3.9505 | 1.1913 | 1.1885–1.1936 |
| scale_14_rank / scale_14_group | 6.2401 | 5.4899 | 1.1366 | 1.1356–1.1389 |
| scale_50_rank / scale_50_group | 20.0820 | 19.3521 | 1.0377 | 1.0371–1.0387 |
| scale_62_rank / scale_62_group | 24.6976 | 23.9618 | 1.0307 | 1.0298–1.0316 |

The ratio describes isolated copy makespan, not model TPOT. Thirty randomized paired repeat blocks support a 10,000-draw bootstrap interval for the ratio of medians. Per-rank CUDA event service and host completion, group bandwidth, p10/p90/p99 and all raw rows are separate artifacts. Group bandwidth is decimal GB/s from common release to that group’s last active rank.

Directional peer costs use measured two-rank NCCL broadcast service, 9-MiB-minus-4-KiB median transfer slopes, normalized to integer scale 1000. This calibration contains only sender and receiver; it is not a full all-to-all latency model. Across groups NCCL selected SHM; within groups P2P/CUMEM. Those transport differences must remain visible in the topology-weighted policy interpretation.

Earlier G2_microbench_attempt1 failed before timing because CUDA host registration had no current context. G2_microbench_attempt2 completed samples but failed in PyTorch 2.5 P2P communicator teardown under per-rank single-GPU visibility. G2_microbench_attempt3 replaced calibration P2P calls with two-rank collectives, completed all measurements, unregistered sources, destroyed groups, and exited cleanly. Those earlier attempts remain under the raw output root; they are not failures of this confirmation job.

Raw root: `/data2/esjung/mgo-results/pcie_topology_ablation_20261009/G2_full54_compact_attempt1`. No model-generation performance is claimed by this experiment.
