# NUMA shared pinned pool size comparison

Effective bandwidth, including solo versus concurrent same-switch per-GPU loss and switch aggregate, is reported in [the bandwidth comparison](../microbench_effective_bandwidth/RESULTS.md). It converts all three raw measurement matrices and includes matched-payload cross-switch controls.

| Source per NUMA | R 4:2 (ms) | G 3:3 (ms) | R/G | H2D latency reduction | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| 144MiB compact | 3.1609 | 2.4177 | 1.3074 | 23.51% | 1.3027–1.3171 |
| 54GiB compact | 3.1671 | 2.4066 | 1.3160 | 24.01% | 1.3106–1.3192 |
| 54GiB spread | 3.1682 | 2.4137 | 1.3126 | 23.82% | 1.3075–1.3176 |

Each full-size run has two physically unique 54-GiB pools (108 GiB total), shared by GPUs0/1 and4/5 respectively. CUDA pinning, same inodes, local page placement, correctness warmups and unregister receipts are checked.

Compact keeps a 144-MiB active source region; spread rotates payload locations throughout the 54-GiB pool. All41 conditions retain30timed samples after5warmups. The complete CSV reports every cell, p10/p90/p99 and independent-job uncertainty.

Between-job differences include temporal drift; they do not establish a causal model speedup. The within-job R/G intervals use paired repeat-block bootstrap. Remote controls in full-size jobs run after the local matrix, registering the other node outside main timings. Both local pools are already physically resident and pinned by their own GPU pairs.

The synthetic transfer path matches full pinned source allocation and registration, but contents are synthetic. Model generation is reported separately. The original frozen G-NUMA-CA peer calibration is retained.
