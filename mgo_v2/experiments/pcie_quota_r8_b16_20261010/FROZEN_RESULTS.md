# Matched-token continuation control

Same R8/C30/B16/input512/output64 Qwen ShareGPT requests, cache budget, Ready-First native executor, Near placement and prefetch OFF. All policies consume the same precomputed next-token IDs from the original Near target. Actual argmax outputs are still computed and recorded. This controls generated-token input drift but BF16 internal hidden states and router choices can differ.

| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Actual token differences vs teacher |
|---|---:|---:|---:|---:|---:|---:|
| Original Near | 3 | 3.102 [1.846, 3.131] | 0.3881 [0.3839, 0.3942] | 27.579 [26.032, 27.935] | 297.041 [293.254, 314.692] | 0 |
| Fast-rank quota | 3 | 3.054 [1.809, 3.068] | 0.3839 [0.3831, 0.3879] | 27.239 [25.946, 27.506] | 300.748 [297.824, 315.727] | 107 |
| PCIe lookup quota | 3 | 3.082 [1.823, 3.175] | 0.3913 [0.3867, 0.3956] | 27.734 [26.184, 28.100] | 295.377 [291.526, 312.869] | 109 |

The table reports the mean of two or median of three unfiltered repeats, with the full range. The token-difference count is from the first target and does not change the common next-token inputs. Per-repeat token differences, rank H2D copy counts, cache state and raw paths are in [FROZEN_RESULTS.json](FROZEN_RESULTS.json).
