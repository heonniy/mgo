# Main-table progress snapshot

Generated UTC 2026-10-06T21:26:39.501434+00:00. This is an incomplete comparison, not the final headline table. All eight cells have completed full primary triplets; two are stable and selected. Five bounded confirmation jobs remain running/queued. OURS small-cell bounded confirmation has completed and remains unstable; no further automatic repeat is scheduled.

Scope: R4, physical GPUs0/1/4/5, C30 expert budget1843 slots (16.1982421875GiB global), identical frozen ShareGPT-long manifests, output64. B labels are requests per OURS/DeepSpeed rank; global batch is4x B. OURS=LA_CA_NEAR, H0/full-pinned.

## Measured triplets

All numbers are seconds. Each displayed estimate is the median of its full three-sample attempt. UNSTABLE means at least one TTFT/TPOT/E2E spread exceeds5%; do not interpret these rows as stability-qualified gains. No repeat is removed as noise. Older and superseded attempts remain archived.

| Cell | System | TTFT | TPOT | E2E | TTFT / TPOT / E2E spread | Status |
| --- | --- | ---: | ---: | ---: | --- | --- |
| B16 / L256 | OURS Near | 4.042 | 0.769798 | 52.539 | 39.51% / 3.76% / 6.86% | UNSTABLE |
| B16 / L256 | MoE-Infinity repaired | 4.916 | 3.050297 | 197.081 | 6.73% / 2.02% / 2.14% | UNSTABLE |
| B16 / L256 | DeepSpeed CPU offload | 5.462 | 4.128344 | 265.502 | 11.94% / 16.02% / 15.92% | UNSTABLE |
| B16 / L256 | llama.cpp static | 37.411 | 0.255360 | 54.235 | 14.59% / 6.16% / 9.75% | UNSTABLE |
| B64 / L512 | OURS Near | 28.007 | 1.415347 | 117.174 | 4.67% / 9.87% / 8.59% | UNSTABLE |
| B64 / L512 | MoE-Infinity repaired | 9.650 | 3.696202 | 242.535 | 1.57% / 3.36% / 3.17% | PASS |
| B64 / L512 | DeepSpeed CPU offload | 5.795 | 4.728005 | 303.659 | 42.24% / 7.69% / 8.43% | UNSTABLE |
| B64 / L512 | llama.cpp static | 334.629 | 0.404266 | 360.068 | 3.45% / 2.86% / 3.01% | PASS |

## Resource observations

GiB=2^30 bytes. HBM is the largest per-GPU1Hz NVML observation during the measured repeats and can miss shorter peaks. Host RSS is measured at repeat end; for OURS/DeepSpeed it is the sum of four process RSS values and may double-count shared pages. It is not unique host physical memory. Pinned columns are observed backing allocations, not an assertion that other implementations use zero pinned memory.

| Cell | System | Max HBM/GPU GiB | Host RSS GiB | Known pinned GiB |
| --- | --- | ---: | ---: | ---: |
| B16 / L256 | OURS Near | 14.22 | 606.39 | 216.00 |
| B16 / L256 | MoE-Infinity repaired | 11.80 | 59.66 | unavailable |
| B16 / L256 | DeepSpeed CPU offload | 11.86 | 90.48 | 56.87 |
| B16 / L256 | llama.cpp static | 14.84 | 41.35 | unavailable |
| B64 / L512 | OURS Near | 19.23 | 677.32 | 216.00 |
| B64 / L512 | MoE-Infinity repaired | 59.95 | 59.82 | unavailable |
| B64 / L512 | DeepSpeed CPU offload | 16.56 | 90.61 | 56.87 |
| B64 / L512 | llama.cpp static | 18.37 | 41.95 | unavailable |

Total HBM includes dense weights, activations, KV and workspaces. The common limit is expert residency, not total30GiB HBM. llama.cpp uses14 GPU expert layers/15.75GiB, conservatively below16.198GiB; host expert operation offload is disabled. DeepSpeed limits all-parameter residency, a conservative upper bound on experts.

## Provenance and limitations

BF16 weights are used. llama.cpp has native FP16 KV and small F32 GGUF tensors; other framework/runtime versions differ as recorded in [BASELINE_VERSIONS.json](BASELINE_VERSIONS.json). This is a whole-system comparison, not isolated placement-policy attribution. MoE-Infinity is explicitly repaired; DeepSpeed is stock ZeRO3 CPU parameter offload, not the paper FastGen baseline.

The [audit snapshot](MAIN_TABLE_PROGRESS_AUDIT.json) contains every completed attempt, all raw sample values/ranges/CVs, invalid-attempt reasons, and only the two explicitly selected stable rows. Its raw paths identify original receipts. Frozen workload and common timing checks are implemented in `mgo_v2/scripts/summarize_headline_r4.py`.

Displayed attempt mapping:

- B16 / L256, OURS Near: `ours_B16_L256_confirmation2`.
- B16 / L256, MoE-Infinity repaired: `infinity_B16_L256_primary4`.
- B16 / L256, DeepSpeed CPU offload: `deepspeed_B16_L256_primary1`.
- B16 / L256, llama.cpp static: `llama_B16_L256_static1`.
- B64 / L512, OURS Near: `ours_B64_L512_primary2`.
- B64 / L512, MoE-Infinity repaired: `infinity_B64_L512_primary3`.
- B64 / L512, DeepSpeed CPU offload: `deepspeed_B64_L512_primary1`.
- B64 / L512, llama.cpp static: `llama_B64_L512_static1`.

OURS small cell: separate launches reproduce a ~45–47% first-primary TTFT premium over the later-two mean. Cold expert-cache and no-compile guards pass, so these observations do not identify the cause. The first sample is retained. The current bounded confirmation median TPOT0.769798s passes its own spread check, but TTFT/E2E keep the whole row unstable.
