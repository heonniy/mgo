# Effective PCIe H2D bandwidth

Primary evidence: two physical 54-GiB NUMA-local shared pinned sources, spread addresses. GPUs 0/1 share one PCIe group; GPUs 4/5 share the other. Existing measurements are converted; no new GPU timing run was performed.

**Units and endpoints.** Decimal GB/s = bytes / seconds / 1e9. Per-GPU CUDA service bandwidth uses the recorded start/end CUDA events on its single copy stream. Switch and system bandwidth use the common CPU release to the last active GPU ready in that scope, including host scheduling and launch/completion overhead. Switch aggregate is bytes actually transferred on that switch divided by that window; it is not the sum of independent per-GPU service rates.

**Payload control.** Each fetch is 9,437,184 bytes (9 MiB). Solo transfers are two sequential fetches on one GPU (18 MiB); same-switch transfers are one fetch on each of two GPUs (9 MiB/GPU, 18 MiB/switch). The solo comparison therefore has a per-GPU payload-count difference. Cross-switch controls use one 9-MiB fetch per GPU, on separate switches, and give the matched-per-GPU-payload comparison. This is synthetic BF16 H2D, not measured generation throughput or a PCIe hardware link-rate claim.

| Pinned source / GPU | Solo service GB/s | Same-switch service GB/s | Drop vs solo | Cross-switch service GB/s | Drop vs matched cross-switch |
|---|---:|---:|---:|---:|---:|
| 144MiB compact / 0 | 11.904 | 6.012 | 49.49% [49.42, 49.57] | 11.560 | 47.99% [47.75, 48.22] |
| 144MiB compact / 1 | 11.934 | 6.026 | 49.50% [49.36, 49.73] | 11.591 | 48.01% [47.73, 48.29] |
| 144MiB compact / 4 | 11.843 | 5.953 | 49.74% [49.54, 49.91] | 11.430 | 47.92% [47.73, 48.20] |
| 144MiB compact / 5 | 11.857 | 5.952 | 49.80% [49.65, 49.99] | 11.529 | 48.37% [47.96, 48.56] |
| 54GiB compact / 0 | 11.874 | 5.988 | 49.57% [49.44, 49.67] | 11.600 | 48.38% [48.15, 48.53] |
| 54GiB compact / 1 | 11.889 | 5.991 | 49.61% [49.48, 49.71] | 11.612 | 48.41% [48.24, 48.51] |
| 54GiB compact / 4 | 11.862 | 5.977 | 49.61% [49.39, 49.81] | 11.446 | 47.78% [47.42, 48.21] |
| 54GiB compact / 5 | 11.867 | 5.979 | 49.62% [49.34, 49.80] | 11.546 | 48.22% [47.87, 48.41] |
| 54GiB spread / 0 | 11.859 | 5.980 | 49.57% [49.32, 49.74] | 11.557 | 48.26% [48.00, 48.48] |
| 54GiB spread / 1 | 11.883 | 5.979 | 49.69% [49.30, 49.81] | 11.566 | 48.31% [47.91, 48.54] |
| 54GiB spread / 4 | 11.886 | 5.955 | 49.90% [49.65, 50.05] | 11.471 | 48.09% [47.65, 48.35] |
| 54GiB spread / 5 | 11.891 | 5.961 | 49.87% [49.62, 50.13] | 11.533 | 48.31% [48.00, 48.59] |

Values are medians of per-repeat bandwidth. Brackets are 95% paired repeat-block bootstrap intervals for the percentage drop (10,000 draws, seed 20261009), pairing conditions only within the same job. All 30 timed repeats after five warmups are included. No cross-job bootstrap pairing or causal pool-size claim is made.

| Pinned source / switch | Solo first GPU GB/s | Solo second GPU GB/s | Both on same switch GB/s | Two separate switches: system GB/s |
|---|---:|---:|---:|---:|
| 144MiB compact / switch_0_1 | 11.530 | 11.629 | 11.616 | 21.269 |
| 144MiB compact / switch_4_5 | 11.427 | 11.484 | 11.471 | 21.269 |
| 54GiB compact / switch_0_1 | 11.542 | 11.558 | 11.610 | 21.400 |
| 54GiB compact / switch_4_5 | 11.451 | 11.428 | 11.498 | 21.400 |
| 54GiB spread / switch_0_1 | 11.454 | 11.500 | 11.561 | 21.464 |
| 54GiB spread / switch_4_5 | 11.526 | 11.511 | 11.458 | 21.464 |

| Pinned source | Six-fetch R 4:2 system GB/s | Six-fetch G 3:3 system GB/s | Effective bandwidth increase |
|---|---:|---:|---:|
| 144MiB compact | 17.913 | 23.421 | 30.74% |
| 54GiB compact | 17.879 | 23.528 | 31.60% |
| 54GiB spread | 17.872 | 23.459 | 31.26% |

Six-fetch values divide the identical 56,623,104-byte payload by the common-release-to-last-ready window. Bandwidth increase is G/R minus one; it differs numerically from latency reduction (one minus R/G bandwidth). These are isolated H2D results, not generation TPOT improvements.

The same-switch per-GPU loss with approximately preserved switch aggregate is consistent with sharing a group bottleneck. The matched cross-switch controls support this interpretation despite the solo payload-count difference. These measurements do not isolate whether the limiting shared resource is the switch upstream link, root port, or another shared path component.

The CSV/JSON artifacts also convert every original microbench condition, including six-fetch placement, scaling, remote-source and overlap sensitivity. Those controls remain explicitly labelled. Multi-stream cells have completion-window bandwidth only: their recorded maximum individual stream duration does not define a valid common CUDA service window.

Pinning, physical sharing, sampled NUMA page locality, repeat counts and original summary timing/bandwidth agreement are validated against the retained raw rows and host receipts. Input SHA256 digests are in bandwidth.json. See [the pinned pool comparison](../microbench_pool_comparison/RESULTS.md) for full source allocation and layout details.

Reproduce from repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data2/esjung/envs/mgo-pcie/bin/python mgo_v2/scripts/pcie_bandwidth_report.py \
  --small mgo_v2/experiments/pcie_topology_ablation_20261009/microbench_physical_cores --compact mgo_v2/experiments/pcie_topology_ablation_20261009/microbench_full54_compact --spread mgo_v2/experiments/pcie_topology_ablation_20261009/microbench_full54_spread --out /tmp/pcie-bandwidth-report
```
