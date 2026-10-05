# Policy regime: audited primary timing (Env1)

Scope: R4 physical GPUs 0/1/4/5, local B128, frozen decode64, BF16 V3 P2/T2. C30 and C60 remain independent comparisons.

Two unprofiled repeats per condition passed the <=2% E2E and TPOT stability gate. All samples are retained; no noise exclusions. Reported estimates are their means.

| Cache | Policy | TPOT seconds (full range) | E2E seconds (full range) | TPOT gain vs BR |
|---|---|---:|---:|---:|
| C30 | BR | 1.782411 (1.778172–1.786649) | 124.468 (124.372–124.565) | +0.000% |
| C30 | OLD_CA | 1.853485 (1.850642–1.856328) | 129.395 (129.337–129.454) | -3.988% |
| C30 | FCA | 1.935111 (1.932149–1.938073) | 135.888 (135.878–135.897) | -8.567% |
| C30 | LA_CA | 1.779419 (1.779030–1.779807) | 125.418 (125.214–125.622) | +0.168% |
| C60 | BR | 1.406486 (1.404538–1.408434) | 100.230 (100.050–100.409) | +0.000% |
| C60 | OLD_CA | 1.447930 (1.446378–1.449482) | 103.281 (103.093–103.469) | -2.947% |
| C60 | FCA | 1.542743 (1.541068–1.544419) | 110.582 (110.506–110.658) | -9.688% |
| C60 | LA_CA | 1.398641 (1.398126–1.399157) | 100.824 (100.780–100.869) | +0.558% |

OLD_CA and FCA are slower in the observed means. LA_CA has small positive point estimates; both paired uncertainty intervals include zero, so these runs do not establish a positive LA_CA gain. Stable repetition is not the same as evidence for a small gain.

Mechanism attribution is still pending. Nsight 2024.6 FCA stalled twice at different decode positions; both failed captures remain archived. Separate captures use an isolated 2025.3 installation with device-side CUDA event completion tracing explicitly off. This is a profiler recovery hypothesis, not a confirmed root cause. Primary measurements are unchanged.

Historical same-host Env2 is queued after attribution: NCCL_CUMEM_ENABLE=0, NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, inherited NCCL overrides cleared. Require observed SHM channels before timing. This is not a claim about physically NVLink-free hardware.

Sources: `POLICY_REGIME_TIMING_RESULTS.json` contains every raw timing, paired gates, and input receipt hashes. `POLICY_REGIME_WORKLOAD.json` reconciles actual physical decode copy counts.
