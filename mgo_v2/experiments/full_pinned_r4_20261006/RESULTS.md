# R4 full-model staged versus full-pinned expert sources

R4 GPUs 0,1,4,5; C30; local B128; BR/P2/T2; H1b; BF16; optimized overlap. Frozen teacher-forced routing and existing requests. One prefill output plus 64 decode steps (65 output positions). One correctness pass and exactly two counterbalanced measurements per mode; no sample exclusions.

All timings in seconds. TTFT is local E2E minus decode wall; TPOT uses the existing CUDA-event decode timer /64. Each reported metric independently takes max across ranks, so aggregate TTFT+decode wall need not exactly equal aggregate E2E. This is not the global outer-wall production main-table timing protocol.

| Mode | Repeat | TTFT | TPOT | Decode wall | E2E |
|---|---:|---:|---:|---:|---:|
| STAGED | 1 | 12.146424 | 1.671178 | 106.961591 | 119.102708 |
| STAGED | 2 | 11.586041 | 1.650373 | 105.631873 | 117.213231 |
| FULL_PINNED | 1 | 9.575397 | 0.781980 | 50.053527 | 59.624851 |
| FULL_PINNED | 2 | 10.093610 | 0.780743 | 49.974492 | 60.063045 |

| Metric | Staged estimate | Full-pinned estimate | Reduction | Staged spread | Full-pinned spread |
|---|---:|---:|---:|---:|---:|
| TTFT | 11.866232 | 9.834503 | +17.122% | 4.722% | 5.269% |
| TPOT | 1.660775 | 0.781361 | +52.952% | 1.253% | 0.158% |
| decode_wall | 106.296732 | 50.014010 | +52.949% | 1.251% | 0.158% |
| E2E_wall | 118.157970 | 59.843948 | +49.353% | 1.599% | 0.732% |

Two-sample median equals mean. Spread=(max-min)/mean; values above 5% are unstable, not headline-ready. All samples are preserved without a third or open-ended repeat.

| Rank / GPU | Pinned GiB | Untimed initialization seconds | Peak HBM GiB | Process max RSS GiB |
|---|---:|---:|---:|---:|
| 0 / 0 | 54 | 49.184 | 49.813 | 174.563 |
| 1 / 1 | 54 | 49.184 | 52.633 | 174.595 |
| 2 / 4 | 54 | 49.184 | 42.852 | 174.560 |
| 3 / 5 | 54 | 48.212 | 47.117 | 174.583 |

Pinned memory totals 216 GiB. Both source stores stay alive in the same process; allocation/copy is outside timing. Peak HBM and max RSS are process lifetime high-water marks, not isolated per-mode footprints. RSS includes shared file-backed pages, so summing rank RSS overstates unique host consumption.

H1b graph input/output/weighted-output buffers are persistent during inference, not startup-only scratch. Their recorded sizes span 28.6–36.2 GiB per rank (see graph receipts). The current implementation does not release them when timing begins; both source modes reuse the same graphs. Reducing this GPU memory requires a separate buffer/graph implementation change.

All four ranks passed token equality, canonical cache/controller validation and physical copy/byte/transport equality across both modes and all repetitions. No compilation occurred in primary measurements. Source-path DMA retains the scheduler, prefetch priority and slot hazards; direct mode skips CPU staging-team materialization as specified by e7209506.

Elapsed supervised execution: 875.5 seconds. Host available before/after: 1846.9/1854.7 GiB. Raw records include all 16 measured rank rows, eight correctness receipts, four pinned-store receipts and four graph receipts.
