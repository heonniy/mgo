# Policy regime results: Env1 and historical Env2

R4 GPUs 0/1/4/5; local B128; frozen decode64; BF16; V3 P2/T2; substitution OFF. C30 and C60 use independent BR denominators. The same frozen requests, routes, weights, and teacher tokens are reused.

Env2 is the historical same-host SHM configuration: NCCL_CUMEM_ENABLE=0, NCCL_P2P_DISABLE=1, NCCL_IB_DISABLE=1, with other inherited NCCL overrides cleared. Both Env2 cache groups must show SHM channels before timing. This is not a measurement on physically NVLink-free hardware.

Env1 preceded Env2; environments were not interleaved. Within-environment policy gains are the primary comparison. An absolute Env1-to-Env2 time difference can also include temporal host drift and is not, by itself, an isolated estimate of transport cost.

Primary timing is unprofiled. Two-repeat estimates use the mean; three-repeat estimates use the median. All raw samples and full ranges remain in the timing JSON. A slow observation alone is not a reason to remove it. A positive point estimate is not proof of a gain; paired uncertainty is reported separately.

## Outcome

env1/C30: lowest TPOT point estimate LA_CA (+0.168% vs BR); lowest mean E2E BR; supported positive TPOT gains: none.

env1/C60: lowest TPOT point estimate LA_CA (+0.558% vs BR); lowest mean E2E BR; supported positive TPOT gains: none.

env2/C30: lowest TPOT point estimate LA_CA (+0.638% vs BR); lowest mean E2E BR; supported positive TPOT gains: none.

env2/C60: lowest TPOT point estimate LA_CA (+0.374% vs BR); lowest mean E2E BR; supported positive TPOT gains: none.

The small LA_CA TPOT point estimates are not established improvements. BR has the lowest mean E2E in all four groups. OLD_CA and FCA have worse observed mean TPOT in both cache regimes and both environments.

FCA reduces remote packets by 29.1% at C30 and 21.3% at C60, with identical packet counts across environments. However, both forward and return NCCL residency increase, and eventwise maximum expert GPU time increases even though rank-mean expert kernel time stays similar. This fails the physical communication-benefit gate. The observations support an imbalance/waiting explanation; they do not isolate a sole cause or measure pure wire latency. The slower FCA primary results are not explained by an observed increase in mean controller CPU time in these separate captures.

No single-sample selection, noise exclusion, or extra timing repetition was used to amplify a gain. All 16 conditions stopped at two stable repeats.

## Primary timing

| Environment | Cache | Policy | TPOT s [range] | E2E s [range] | TPOT gain | Paired gain 95% interval |
|---|---|---|---:|---:|---:|---:|
| env1 | C30 | BR | 1.782411 [1.778172, 1.786649] | 124.468 [124.372, 124.565] | +0.000% | baseline |
| env1 | C30 | OLD_CA | 1.853485 [1.850642, 1.856328] | 129.395 [129.337, 129.454] | -3.988% | [-9.29%, +1.05%] |
| env1 | C30 | FCA | 1.935111 [1.932149, 1.938073] | 135.888 [135.878, 135.897] | -8.567% | [-9.74%, -7.40%] |
| env1 | C30 | LA_CA | 1.779419 [1.779030, 1.779807] | 125.418 [125.214, 125.622] | +0.168% | [-2.61%, +2.87%] |
| env1 | C60 | BR | 1.406486 [1.404538, 1.408434] | 100.230 [100.050, 100.409] | +0.000% | baseline |
| env1 | C60 | OLD_CA | 1.447930 [1.446378, 1.449482] | 103.281 [103.093, 103.469] | -2.947% | [-3.36%, -2.54%] |
| env1 | C60 | FCA | 1.542743 [1.541068, 1.544419] | 110.582 [110.506, 110.658] | -9.688% | [-13.19%, -6.30%] |
| env1 | C60 | LA_CA | 1.398641 [1.398126, 1.399157] | 100.824 [100.780, 100.869] | +0.558% | [-0.73%, +1.83%] |
| env2 | C30 | BR | 1.784403 [1.783281, 1.785525] | 125.069 [125.000, 125.138] | +0.000% | baseline |
| env2 | C30 | OLD_CA | 1.847691 [1.847592, 1.847790] | 129.670 [129.616, 129.723] | -3.547% | [-4.45%, -2.65%] |
| env2 | C30 | FCA | 1.926511 [1.926014, 1.927008] | 136.074 [135.998, 136.151] | -7.964% | [-8.47%, -7.46%] |
| env2 | C30 | LA_CA | 1.773027 [1.772060, 1.773995] | 125.344 [125.227, 125.460] | +0.638% | [-0.86%, +2.11%] |
| env2 | C60 | BR | 1.396235 [1.396123, 1.396347] | 100.240 [100.161, 100.319] | +0.000% | baseline |
| env2 | C60 | OLD_CA | 1.434747 [1.433664, 1.435831] | 103.247 [103.145, 103.349] | -2.758% | [-3.64%, -1.88%] |
| env2 | C60 | FCA | 1.533615 [1.531622, 1.535609] | 110.931 [110.839, 111.023] | -9.839% | [-11.78%, -7.93%] |
| env2 | C60 | LA_CA | 1.391008 [1.390587, 1.391430] | 100.878 [100.856, 100.901] | +0.374% | [-0.11%, +0.86%] |

env1/C30: unstable=False; repeat counts: BR=2, OLD_CA=2, FCA=2, LA_CA=2.
env1/C60: unstable=False; repeat counts: BR=2, OLD_CA=2, FCA=2, LA_CA=2.
env2/C30: unstable=False; repeat counts: BR=2, OLD_CA=2, FCA=2, LA_CA=2.
env2/C60: unstable=False; repeat counts: BR=2, OLD_CA=2, FCA=2, LA_CA=2.

## Mechanism attribution

Values below are means across the four rank-local durations per decode step (ms). They come from separate instrumented captures, not primary TPOT. Do not add them: phases can overlap, NCCL includes waiting, and host staging runs on its own CPU thread. Outside-COMM/COMPUTE H2D is an overlap definition, not proof of critical-path stall. One capture per cell is descriptive.

| Env | Cache | Policy | H2D DMA | H2D outside | Forward NCCL | Return NCCL | Expert GPU | Current controller CPU | Prefetch controller CPU | Host staging CPU |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| env1 | C30 | BR | 168.145 | 159.959 | 49.188 | 103.489 | 33.726 | 127.589 | 46.601 | 975.242 |
| env1 | C30 | OLD_CA | 168.321 | 157.216 | 68.559 | 181.753 | 33.680 | 123.098 | 40.706 | 972.012 |
| env1 | C30 | FCA | 168.858 | 155.514 | 76.130 | 272.509 | 33.651 | 123.541 | 40.566 | 969.878 |
| env1 | C30 | LA_CA | 168.296 | 161.014 | 50.104 | 105.656 | 33.747 | 133.953 | 40.915 | 970.344 |
| env1 | C60 | BR | 84.876 | 78.708 | 49.203 | 112.114 | 34.178 | 119.257 | 55.606 | 504.619 |
| env1 | C60 | OLD_CA | 87.070 | 78.846 | 57.545 | 150.569 | 34.113 | 113.928 | 50.155 | 504.906 |
| env1 | C60 | FCA | 88.578 | 77.977 | 66.191 | 270.987 | 34.115 | 113.570 | 50.545 | 502.224 |
| env1 | C60 | LA_CA | 84.910 | 79.355 | 41.723 | 109.617 | 34.152 | 119.370 | 50.439 | 500.941 |
| env2 | C30 | BR | 168.571 | 158.693 | 55.918 | 107.104 | 33.667 | 128.250 | 47.077 | 970.548 |
| env2 | C30 | OLD_CA | 168.932 | 156.726 | 73.735 | 182.216 | 33.620 | 122.301 | 40.673 | 966.802 |
| env2 | C30 | FCA | 168.818 | 154.484 | 75.914 | 275.612 | 33.589 | 122.907 | 40.727 | 963.623 |
| env2 | C30 | LA_CA | 168.357 | 159.404 | 57.634 | 88.078 | 33.654 | 133.376 | 41.215 | 972.608 |
| env2 | C60 | BR | 84.941 | 77.111 | 53.209 | 113.404 | 33.669 | 118.352 | 56.117 | 505.713 |
| env2 | C60 | OLD_CA | 86.883 | 77.656 | 63.276 | 181.617 | 33.636 | 113.858 | 49.813 | 505.302 |
| env2 | C60 | FCA | 88.763 | 77.494 | 70.331 | 266.696 | 33.632 | 113.176 | 50.328 | 501.136 |
| env2 | C60 | LA_CA | 85.042 | 78.352 | 49.555 | 105.013 | 33.676 | 119.158 | 50.409 | 500.391 |

## Eventwise imbalance and packet counts

Each duration below sums per-event rank maxima and divides by 64. It is an imbalance proxy, not a critical-path wall time. Rank means can hide a busy rank that changes between events.

| Env | Cache | Policy | Remote packets | Sum max incident packets | Expert max ms/step | Forward max ms/step | Return max ms/step |
|---|---|---|---:|---:|---:|---:|---:|
| env1 | C30 | BR | 4,266,106 | 2,189,850 | 37.852 | 80.349 | 214.560 |
| env1 | C30 | OLD_CA | 3,242,561 | 1,827,738 | 43.365 | 107.200 | 338.689 |
| env1 | C30 | FCA | 3,024,330 | 1,751,846 | 48.419 | 117.161 | 458.070 |
| env1 | C30 | LA_CA | 4,176,873 | 2,127,890 | 36.700 | 78.515 | 218.977 |
| env1 | C60 | BR | 4,248,700 | 2,185,075 | 39.632 | 88.365 | 227.630 |
| env1 | C60 | OLD_CA | 3,663,474 | 1,974,825 | 42.494 | 98.855 | 296.438 |
| env1 | C60 | FCA | 3,343,756 | 1,866,092 | 49.873 | 109.556 | 478.228 |
| env1 | C60 | LA_CA | 4,135,287 | 2,122,567 | 38.939 | 78.258 | 225.801 |
| env2 | C30 | BR | 4,266,106 | 2,189,850 | 37.786 | 84.207 | 217.898 |
| env2 | C30 | OLD_CA | 3,242,561 | 1,827,738 | 43.282 | 109.529 | 336.686 |
| env2 | C30 | FCA | 3,024,330 | 1,751,846 | 48.349 | 112.214 | 462.209 |
| env2 | C30 | LA_CA | 4,176,873 | 2,127,890 | 36.590 | 86.221 | 189.319 |
| env2 | C60 | BR | 4,248,700 | 2,185,075 | 39.026 | 89.078 | 227.518 |
| env2 | C60 | OLD_CA | 3,663,474 | 1,974,825 | 41.878 | 101.864 | 356.572 |
| env2 | C60 | FCA | 3,343,756 | 1,866,092 | 49.138 | 109.338 | 468.283 |
| env2 | C60 | LA_CA | 4,135,287 | 2,122,567 | 38.387 | 83.118 | 215.493 |

Full rank min/max distributions, per-event expert/communication imbalance, remote packet counts, and critical-rank packet load are retained in each `POLICY_REGIME_MECHANISMS.json`. Actual decode copy counts, prefetch usage, and issued-copy precision are in `POLICY_REGIME_WORKLOAD.json`. Prefetch use does not by itself establish readiness or time saved.

Nsight 2024.6 FCA stalled twice. Both failures remain archived. Attribution was recollected with isolated Nsight 2025.3 and device event completion tracing disabled; completed old captures are not mixed into the final duration comparison. Primary timing was not repeated for this recovery.

Machine-readable source receipt hashes: `POLICY_REGIME_FINAL_SOURCES.json`.
