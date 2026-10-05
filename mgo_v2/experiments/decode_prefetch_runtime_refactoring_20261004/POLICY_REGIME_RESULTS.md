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

All eight Env1 mechanism captures and all 32 rank-level DMA count/byte reconciliations passed. Nsight 2024.6 FCA stalled twice; both failed captures remain archived. The final duration comparisons use isolated Nsight 2025.3 with device-side event completion tracing off for every policy. This recovered the full captures, but does not prove the original CUDA deadlock cause. Primary measurements were not repeated for this recovery.

FCA fails the physical communication-benefit gate in both cache groups: remote packets fall by about 29.1% (C30) and 21.3% (C60), but captured forward and return NCCL residency rise and primary TPOT is worse. NCCL residency includes peer readiness and waiting, so these are not pure link-transfer times. Mean expert kernel unions are similar across policies, while per-event maximum rank compute is larger for FCA. This supports an imbalance/waiting mechanism; it does not identify a single causal bottleneck or equate row count with GPU duration.

LA_CA preserves the lower eventwise expert-compute maxima and remains close to BR in communication time. Its small positive primary TPOT estimates are not established gains because both paired intervals include zero. Do not choose an allegedly noise-free sample to amplify these estimates.

The table shows rank-mean, separately instrumented milliseconds per step, except Expert max which is the sum of eventwise rank maxima divided by 64. None of these columns can be added into TPOT. H2D outside is outside the union of metadata/payload NCCL and expert GPU work, an overlap definition rather than proven exposed stall.

| Cache | Policy | Remote packets | H2D DMA ms | H2D outside ms | Forward NCCL ms | Return NCCL ms | Expert max ms |
|---|---|---:|---:|---:|---:|---:|---:|
| C30 | BR | 4,266,106 | 168.145 | 159.959 | 49.188 | 103.489 | 37.852 |
| C30 | OLD_CA | 3,242,561 | 168.321 | 157.216 | 68.559 | 181.753 | 43.365 |
| C30 | FCA | 3,024,330 | 168.858 | 155.514 | 76.130 | 272.509 | 48.419 |
| C30 | LA_CA | 4,176,873 | 168.296 | 161.014 | 50.104 | 105.656 | 36.700 |
| C60 | BR | 4,248,700 | 84.876 | 78.708 | 49.203 | 112.114 | 39.632 |
| C60 | OLD_CA | 3,663,474 | 87.070 | 78.846 | 57.545 | 150.569 | 42.494 |
| C60 | FCA | 3,343,756 | 88.578 | 77.977 | 66.191 | 270.987 | 49.873 |
| C60 | LA_CA | 4,135,287 | 84.910 | 79.355 | 41.723 | 109.617 | 38.939 |


Historical same-host Env2 has now started after successful attribution: NCCL_CUMEM_ENABLE=0, NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, inherited NCCL overrides cleared. Require observed SHM channels before timing. This is not a claim about physically NVLink-free hardware.

Sources: `POLICY_REGIME_TIMING_RESULTS.json` contains every raw timing, paired gates, and input receipt hashes. `POLICY_REGIME_WORKLOAD.json` reconciles actual physical decode copy counts.
