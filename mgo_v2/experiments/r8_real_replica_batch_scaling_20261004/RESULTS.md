# R8 B128/B256 phase-aware policy CPU result

**Overall engineering gate: STRONG_GO**

The overall gate is carried by **LA without replication** in all four cases;
it is not a GO for persistent replication. The selected replica variants
increase H2D by 6.67–9.40% versus their corresponding non-replica baseline,
exceeding both the 2% GO cap and the 5% MARGINAL cap. Under the stated gate,
these selected replica comparisons are NO_GO despite modeled phase savings.
LA+REP adds only 0.98–2.43% robust phase savings over LA.

These percentages are calibrated CPU MoE phase-model estimates, not measured
full-model E2E or TPOT gains. Four-GPU pair calibration does not reproduce
R8 collective contention. The COMM batch-scaling gate passes (+2.13 percentage
points); the LOAD gate does not (+0.93 percentage points).

Stress workloads are optimized headroom cases, not dataset averages.

## Stress winners

| Batch | Stress | Dataset | sample / DP / placement | BR peer GiB | BR critical rows |
|---:|---|---|---|---:|---:|
| 128 | COMM | ShareGPT | 55 / 42 / 165 | 559.27 | 18126472 |
| 128 | LOAD | MATH | 64 / 251 / 241 | 557.21 | 20128324 |
| 256 | COMM | MATH | 0 / 8 / 245 | 1120.83 | 37584288 |
| 256 | LOAD | MATH | 0 / 163 / 159 | 1115.04 | 39807631 |

## Selected policy comparison

| B | Stress | Policy | Critical rows | H2D TiB | D2D GiB | Peer GiB | E1 ovlp | E1 serial | E2 ovlp | E2 serial |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | COMM | BR | 18126472 | 5.496 | 0.000 | 559.268 | 208215.1 | 208215.1 | 211117.0 | 211117.0 |
| 128 | COMM | BR+REP | 16060580 | 5.901 | 2.074 | 560.005 | 187820.3 | 187839.3 | 190813.7 | 190894.3 |
| 128 | COMM | CA | 20067506 | 5.502 | 0.000 | 530.370 | 228563.6 | 228563.6 | 231515.0 | 231515.0 |
| 128 | COMM | CA+REP | 17250497 | 5.908 | 1.960 | 534.903 | 200303.2 | 200321.1 | 203318.4 | 203394.4 |
| 128 | COMM | LA | 13331440 | 5.496 | 0.000 | 561.928 | 158001.1 | 158001.1 | 160787.4 | 160787.4 |
| 128 | COMM | LA+REP | 13055398 | 5.862 | 1.696 | 560.303 | 156224.1 | 156239.5 | 159141.7 | 159207.7 |
| 128 | LOAD | BR | 20128324 | 4.117 | 0.000 | 557.207 | 225106.6 | 225106.6 | 227621.0 | 227621.0 |
| 128 | LOAD | BR+REP | 16910969 | 4.504 | 2.716 | 559.947 | 192568.4 | 192593.2 | 195143.1 | 195247.6 |
| 128 | LOAD | CA | 21026221 | 4.130 | 0.000 | 539.347 | 234551.3 | 234551.3 | 237089.2 | 237089.2 |
| 128 | LOAD | CA+REP | 17557435 | 4.519 | 2.654 | 542.073 | 199381.8 | 199406.0 | 201969.8 | 202072.5 |
| 128 | LOAD | LA | 13782316 | 4.117 | 0.000 | 565.760 | 158644.1 | 158644.1 | 161000.5 | 161000.5 |
| 128 | LOAD | LA+REP | 13281115 | 4.488 | 2.355 | 562.600 | 154506.2 | 154527.8 | 157002.1 | 157093.4 |
| 256 | COMM | BR | 37584288 | 4.428 | 0.000 | 1120.832 | 408868.5 | 408868.5 | 412173.8 | 412173.8 |
| 256 | COMM | BR+REP | 33099434 | 4.826 | 2.575 | 1122.217 | 363097.7 | 363121.2 | 366449.1 | 366548.7 |
| 256 | COMM | CA | 42671645 | 4.441 | 0.000 | 1083.607 | 462202.4 | 462202.4 | 465635.3 | 465635.3 |
| 256 | COMM | CA+REP | 35092060 | 4.840 | 2.566 | 1090.473 | 384006.2 | 384029.6 | 387384.2 | 387483.8 |
| 256 | COMM | LA | 27391175 | 4.428 | 0.000 | 1130.203 | 302119.9 | 302119.9 | 305122.6 | 305122.6 |
| 256 | COMM | LA+REP | 26549464 | 4.827 | 2.549 | 1124.634 | 294498.1 | 294521.4 | 297700.5 | 297799.3 |
| 256 | LOAD | BR | 39807631 | 4.428 | 0.000 | 1115.040 | 432148.0 | 432148.0 | 435515.6 | 435515.6 |
| 256 | LOAD | BR+REP | 33665906 | 4.828 | 2.777 | 1120.593 | 369018.6 | 369044.0 | 372373.7 | 372481.4 |
| 256 | LOAD | CA | 40835558 | 4.441 | 0.000 | 1087.607 | 442956.8 | 442956.8 | 446339.4 | 446339.4 |
| 256 | LOAD | CA+REP | 34910604 | 4.841 | 2.698 | 1093.309 | 382103.1 | 382127.7 | 385489.7 | 385594.4 |
| 256 | LOAD | LA | 27285055 | 4.429 | 0.000 | 1129.203 | 301004.3 | 301004.3 | 304004.7 | 304004.7 |
| 256 | LOAD | LA+REP | 26504631 | 4.828 | 2.610 | 1125.991 | 294020.1 | 294043.9 | 297218.3 | 297319.2 |

## GO decisions

- B128_COMM: **STRONG_GO**, best=LA, robust phase gain=23.84%, critical-row gain=26.45%, H2D growth=-0.00%.
- B128_LOAD: **STRONG_GO**, best=LA, robust phase gain=29.27%, critical-row gain=31.53%, H2D growth=-0.01%.
- B256_COMM: **STRONG_GO**, best=LA, robust phase gain=25.97%, critical-row gain=27.12%, H2D growth=0.01%.
- B256_LOAD: **STRONG_GO**, best=LA, robust phase gain=30.20%, critical-row gain=31.46%, H2D growth=0.04%.

## Batch scaling

- COMM: B128 23.84% -> B256 25.97%; delta 2.13%; ratio 1.089458634515036; supports=True.
- LOAD: B128 29.27% -> B256 30.20%; delta 0.93%; ratio 1.0317211965887545; supports=False.

GO uses decode only and requires robustness across Env1/Env2 and both D2D-overlap and serial counterfactual phase models.
Prefill is reported separately as MoE-only headroom; no TTFT claim is made without physical full-model timing.
