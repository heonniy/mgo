# R4 full-model staged versus full-pinned expert sources

R4 GPUs 0,1,4,5; C30; local B128; BR/P2/T2; H0; BF16; optimized overlap. Frozen teacher-forced routing and existing requests. One prefill output plus 64 decode steps (65 output positions). One correctness pass and exactly two counterbalanced measurements per mode; no sample exclusions.

Existing frozen prompts have 16–512 valid tokens; rank-local padded lengths are [302, 512, 335, 406]. TTFT applies to this mixed-length workload.

All timings in seconds. TTFT is local E2E minus decode wall; TPOT uses the existing CUDA-event decode timer /64. Each reported metric independently takes max across ranks, so aggregate TTFT+decode wall need not exactly equal aggregate E2E. This is not the global outer-wall production main-table timing protocol.

| Mode | Repeat | TTFT | TPOT | Decode wall | E2E |
|---|---:|---:|---:|---:|---:|
| STAGED | 1 | 10.266890 | 1.799452 | 115.164875 | 125.426887 |
| STAGED | 2 | 10.281257 | 1.786862 | 114.359185 | 124.637493 |
| FULL_PINNED | 1 | 8.561531 | 1.004953 | 64.317219 | 72.878490 |
| FULL_PINNED | 2 | 8.583994 | 1.037217 | 66.381985 | 74.963737 |

| Metric | Staged estimate | Full-pinned estimate | Reduction | Staged spread | Full-pinned spread |
|---|---:|---:|---:|---:|---:|
| TTFT | 10.274073 | 8.572762 | +16.559% | 0.140% | 0.262% |
| TPOT | 1.793157 | 1.021085 | +43.057% | 0.702% | 3.160% |
| decode_wall | 114.762030 | 65.349602 | +43.056% | 0.702% | 3.160% |
| E2E_wall | 125.032190 | 73.921114 | +40.878% | 0.631% | 2.821% |

Two-sample median equals mean. Spread=(max-min)/mean; values above 5% are unstable, not headline-ready. All samples are preserved without a third or open-ended repeat.

| Rank / GPU | Pinned GiB | Untimed initialization seconds | Peak PyTorch-allocated GPU GiB | Process max RSS GiB |
|---|---:|---:|---:|---:|
| 0 / 0 | 54 | 45.853 | 12.461 | 174.419 |
| 1 / 1 | 54 | 46.440 | 16.247 | 172.576 |
| 2 / 4 | 54 | 40.031 | 13.046 | 172.609 |
| 3 / 5 | 54 | 39.721 | 14.333 | 174.437 |

Pinned memory totals 216 GiB. Both source stores stay alive in the same process; allocation/copy is outside timing. GPU peaks are PyTorch allocated-byte high-water marks, excluding driver/NCCL and reserved-but-unused allocator memory. GPU peaks and max RSS are process lifetime high-water marks, not isolated per-mode footprints. RSS includes shared file-backed pages, so summing rank RSS overstates unique host consumption.

Normal/H0 is used throughout both modes. There are zero GraphExpertExecutor entries and zero persistent graph input/output buffers. The same full-pinned factory used by the selected runtime was physically exercised.

All four ranks passed token equality, canonical cache/controller validation and physical copy/byte/transport equality across both modes and all repetitions. No compilation occurred in primary measurements. Source-path DMA retains the scheduler, prefetch priority and slot hazards; direct mode skips CPU staging-team materialization as specified by e7209506.

Elapsed supervised execution: 762.6 seconds. Host available before/after: 1849.7/1854.5 GiB. Raw records include all 16 measured rank rows, eight correctness receipts, four pinned-store receipts and four executor/graph receipts.
