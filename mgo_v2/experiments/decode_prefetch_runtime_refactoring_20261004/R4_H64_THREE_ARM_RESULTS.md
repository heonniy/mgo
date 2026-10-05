# BF16 three-arm paired timing

Status: NO_STABLE_CANDIDATE. Timing candidate: None. Positive LA gain supported in both batches: False.

BF16 common-stack V1/V2/V3 only; final selection pending profiling and secondary CA

| Arm | Batch | BR TPOT (s) | LA TPOT (s) | Paired LA gain | 95% interval | Repeats | Eligible |
|---|---:|---:|---:|---:|---|---:|---|
| V1_OPT_NOPF_BARRIER | 128 | 1.925220 | 1.933103 | -0.42% | [-5.72%, 4.62%] | 3 | False |
| V1_OPT_NOPF_BARRIER | 256 | 2.205044 | 2.164017 | 1.88% | [-9.95%, 12.44%] | 3 | False |
| V2_OPT_PF_BARRIER | 128 | 1.826746 | 1.821545 | 0.29% | [-2.39%, 2.89%] | 2 | False |
| V2_OPT_PF_BARRIER | 256 | 2.120772 | 2.127604 | -0.32% | [-4.01%, 3.23%] | 3 | False |
| V3_OPT_PF_OVERLAP | 128 | 1.484938 | 1.442529 | 2.82% | [-4.59%, 9.70%] | 3 | False |
| V3_OPT_PF_OVERLAP | 256 | 1.639496 | 1.604878 | 2.11% | [-13.12%, 15.29%] | 2 | False |

| Arm | Batch | BR E2E (s) | LA E2E (s) | Paired E2E gain | 95% interval |
|---|---:|---:|---:|---:|---|
| V1_OPT_NOPF_BARRIER | 128 | 134.480058 | 135.003186 | -0.39% | [-5.45%, 4.42%] |
| V1_OPT_NOPF_BARRIER | 256 | 159.636718 | 156.791079 | 1.80% | [-8.51%, 11.13%] |
| V2_OPT_PF_BARRIER | 128 | 128.276433 | 127.519216 | 0.59% | [-1.50%, 2.64%] |
| V2_OPT_PF_BARRIER | 256 | 154.169665 | 154.552599 | -0.25% | [-3.19%, 2.61%] |
| V3_OPT_PF_OVERLAP | 128 | 106.074431 | 103.456537 | 2.43% | [-4.52%, 8.93%] |
| V3_OPT_PF_OVERLAP | 256 | 123.502106 | 121.688141 | 1.47% | [-16.01%, 16.32%] |

BR/LA absolute times are arithmetic means. Relative gains use paired log ratios and therefore need not equal the ratio of the displayed absolute means. JSON retains E2E, all raw pairs, full ranges, stability decisions, and exclusions.
