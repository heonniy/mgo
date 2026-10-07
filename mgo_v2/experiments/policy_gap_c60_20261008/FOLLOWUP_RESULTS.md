# C30 and P2P-disabled follow-up results

ShareGPT, R4 GPUs0/1/4/5, input128,32 decode forwards, native main_OURS path, prefetch OFF, cold cache after warmup. The four unique seed cases were selected by the best observed C60/NVSwitch clean gain for each policy and batch; B8 has two seeds. Each row has one clean full-TPOT primary (attention included). Diagnostic values come from a separate instrumented run. P2P-disabled sets NCCL_P2P_DISABLE=1 and NCCL_IB_DISABLE=1 inside every worker to select SHM after the IB path failed; it is a transport setting on the same NVSwitch-equipped server.

| Transport | C | B/rank | Seed sample/order | Policy | Clean TPOT ms/token | Diagnostic full decode s | First dispatch→last return s | EP layer windows s | Expert compute s | Dispatch/return s | Arrival/other s | Peer GiB | H2D GiB |
|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nvswitch | 30 | 8 | 20/7 | BR | 447.685 | 15.832 | 15.827 | 10.665 | 6.900 | 3.261 | 0.504 | 1.024 | 576.791 |
| nvswitch | 30 | 8 | 20/7 | CA_NATIVE | 454.254 | 16.172 | 16.167 | 10.783 | 7.239 | 3.135 | 0.409 | 0.866 | 578.874 |
| nvswitch | 30 | 8 | 20/7 | LA_CA_NEAR | 436.421 | 15.654 | 15.650 | 10.362 | 6.800 | 3.116 | 0.446 | 0.985 | 576.932 |
| nvswitch | 30 | 8 | 14/5 | BR | 437.526 | 15.571 | 15.566 | 10.336 | 6.869 | 3.066 | 0.402 | 1.021 | 521.051 |
| nvswitch | 30 | 8 | 14/5 | CA_NATIVE | 443.275 | 15.824 | 15.819 | 10.456 | 6.929 | 3.105 | 0.422 | 0.885 | 523.406 |
| nvswitch | 30 | 8 | 14/5 | LA_CA_NEAR | 429.958 | 15.228 | 15.223 | 9.896 | 6.498 | 3.058 | 0.340 | 0.988 | 521.323 |
| nvswitch | 30 | 16 | 22/3 | BR | 523.985 | 18.134 | 18.129 | 12.553 | 8.817 | 3.291 | 0.445 | 2.043 | 812.074 |
| nvswitch | 30 | 16 | 22/3 | CA_NATIVE | 529.816 | 18.442 | 18.437 | 12.722 | 8.870 | 3.293 | 0.558 | 1.770 | 814.184 |
| nvswitch | 30 | 16 | 22/3 | LA_CA_NEAR | 513.153 | 17.898 | 17.892 | 12.177 | 8.518 | 3.243 | 0.415 | 1.962 | 812.531 |
| nvswitch | 30 | 64 | 20/5 | BR | 598.737 | 20.623 | 20.618 | 14.547 | 10.331 | 3.658 | 0.557 | 8.174 | 1051.708 |
| nvswitch | 30 | 64 | 20/5 | CA_NATIVE | 610.879 | 21.230 | 21.225 | 14.821 | 10.595 | 3.638 | 0.588 | 7.472 | 1052.745 |
| nvswitch | 30 | 64 | 20/5 | LA_CA_NEAR | 586.181 | 20.599 | 20.594 | 14.357 | 10.350 | 3.485 | 0.522 | 8.009 | 1051.497 |
| p2p_disabled | 30 | 8 | 20/7 | BR | 451.557 | 16.014 | 16.009 | 10.749 | 6.956 | 3.333 | 0.460 | 1.024 | 576.791 |
| p2p_disabled | 30 | 8 | 20/7 | CA_NATIVE | 458.144 | 16.233 | 16.228 | 10.800 | 7.251 | 3.128 | 0.422 | 0.866 | 578.874 |
| p2p_disabled | 30 | 8 | 20/7 | LA_CA_NEAR | 444.598 | 15.702 | 15.697 | 10.294 | 6.839 | 3.085 | 0.370 | 0.985 | 576.932 |
| p2p_disabled | 30 | 8 | 14/5 | BR | 436.450 | 15.467 | 15.458 | 10.291 | 6.703 | 3.125 | 0.463 | 1.021 | 521.051 |
| p2p_disabled | 30 | 8 | 14/5 | CA_NATIVE | 443.018 | 15.690 | 15.684 | 10.428 | 6.408 | 3.515 | 0.505 | 0.885 | 523.406 |
| p2p_disabled | 30 | 8 | 14/5 | LA_CA_NEAR | 425.118 | 15.172 | 15.166 | 9.873 | 6.368 | 3.099 | 0.406 | 0.988 | 521.323 |
| p2p_disabled | 30 | 16 | 22/3 | BR | 523.418 | 18.217 | 18.211 | 12.514 | 8.868 | 3.245 | 0.401 | 2.043 | 812.074 |
| p2p_disabled | 30 | 16 | 22/3 | CA_NATIVE | 529.840 | 18.507 | 18.501 | 12.578 | 8.860 | 3.289 | 0.429 | 1.770 | 814.184 |
| p2p_disabled | 30 | 16 | 22/3 | LA_CA_NEAR | 512.613 | 17.972 | 17.967 | 12.203 | 8.482 | 3.259 | 0.461 | 1.962 | 812.531 |
| p2p_disabled | 30 | 64 | 20/5 | BR | 594.726 | 20.498 | 20.493 | 14.380 | 10.497 | 3.392 | 0.490 | 8.174 | 1051.708 |
| p2p_disabled | 30 | 64 | 20/5 | CA_NATIVE | 604.834 | 20.936 | 20.931 | 14.575 | 10.583 | 3.421 | 0.570 | 7.472 | 1052.745 |
| p2p_disabled | 30 | 64 | 20/5 | LA_CA_NEAR | 585.696 | 20.272 | 20.266 | 14.064 | 10.174 | 3.418 | 0.472 | 8.009 | 1051.497 |
| p2p_disabled | 60 | 8 | 20/7 | BR | 432.776 | 15.340 | 15.335 | 10.002 | 6.641 | 2.947 | 0.414 | 1.026 | 204.346 |
| p2p_disabled | 60 | 8 | 20/7 | CA_NATIVE | 443.237 | 15.775 | 15.769 | 10.330 | 7.178 | 2.767 | 0.386 | 0.878 | 205.910 |
| p2p_disabled | 60 | 8 | 20/7 | LA_CA_NEAR | 424.856 | 15.284 | 15.279 | 9.842 | 6.823 | 2.669 | 0.350 | 0.996 | 203.871 |
| p2p_disabled | 60 | 8 | 14/5 | BR | 416.894 | 15.131 | 15.127 | 9.784 | 6.440 | 2.928 | 0.415 | 1.024 | 160.497 |
| p2p_disabled | 60 | 8 | 14/5 | CA_NATIVE | 429.452 | 15.652 | 15.647 | 10.145 | 7.126 | 2.708 | 0.311 | 0.894 | 162.343 |
| p2p_disabled | 60 | 8 | 14/5 | LA_CA_NEAR | 417.191 | 14.922 | 14.917 | 9.555 | 6.446 | 2.693 | 0.415 | 0.990 | 159.917 |
| p2p_disabled | 60 | 16 | 22/3 | BR | 510.141 | 17.922 | 17.917 | 12.107 | 8.533 | 3.166 | 0.407 | 2.050 | 370.222 |
| p2p_disabled | 60 | 16 | 22/3 | CA_NATIVE | 524.148 | 18.364 | 18.359 | 12.432 | 8.780 | 3.225 | 0.427 | 1.804 | 371.259 |
| p2p_disabled | 60 | 16 | 22/3 | LA_CA_NEAR | 510.896 | 18.011 | 18.006 | 12.101 | 8.209 | 3.410 | 0.482 | 1.966 | 370.090 |
| p2p_disabled | 60 | 64 | 20/5 | BR | 575.721 | 20.345 | 20.339 | 14.047 | 10.116 | 3.447 | 0.484 | 8.219 | 543.771 |
| p2p_disabled | 60 | 64 | 20/5 | CA_NATIVE | 583.911 | 20.776 | 20.770 | 14.336 | 10.343 | 3.466 | 0.527 | 7.438 | 545.177 |
| p2p_disabled | 60 | 64 | 20/5 | LA_CA_NEAR | 577.902 | 20.388 | 20.378 | 13.999 | 9.748 | 3.721 | 0.530 | 8.010 | 544.263 |

## Paired clean TPOT comparison

Positive gain means faster than BR on the same seed. EP-window delta uses the separate diagnostic run and is not a clean-TPOT decomposition.

| Transport | C | B/rank | Seed | BR TPOT ms/token | CA_NATIVE gain % | Near gain % | Near−BR EP windows s/32 steps |
|---|---:|---:|---|---:|---:|---:|---:|
| nvswitch | 30 | 8 | 20/7 | 447.685 | -1.47 | +2.52 | -0.303 |
| nvswitch | 30 | 8 | 14/5 | 437.526 | -1.31 | +1.73 | -0.441 |
| nvswitch | 30 | 16 | 22/3 | 523.985 | -1.11 | +2.07 | -0.377 |
| nvswitch | 30 | 64 | 20/5 | 598.737 | -2.03 | +2.10 | -0.189 |
| p2p_disabled | 30 | 8 | 20/7 | 451.557 | -1.46 | +1.54 | -0.455 |
| p2p_disabled | 30 | 8 | 14/5 | 436.450 | -1.50 | +2.60 | -0.418 |
| p2p_disabled | 30 | 16 | 22/3 | 523.418 | -1.23 | +2.06 | -0.311 |
| p2p_disabled | 30 | 64 | 20/5 | 594.726 | -1.70 | +1.52 | -0.316 |
| p2p_disabled | 60 | 8 | 20/7 | 432.776 | -2.42 | +1.83 | -0.160 |
| p2p_disabled | 60 | 8 | 14/5 | 416.894 | -3.01 | -0.07 | -0.229 |
| p2p_disabled | 60 | 16 | 22/3 | 510.141 | -2.75 | -0.15 | -0.006 |
| p2p_disabled | 60 | 64 | 20/5 | 575.721 | -1.42 | -0.38 | -0.048 |

## Interpretation

- `nvswitch` C30: Near observed clean gain +1.73% to +2.52%; CA_NATIVE -2.03% to -1.11% despite 8.6%–15.4% fewer decode peer bytes than BR.
- `p2p_disabled` C30: Near observed clean gain +1.52% to +2.60%; CA_NATIVE -1.70% to -1.23% despite 8.6%–15.4% fewer decode peer bytes than BR.
- `p2p_disabled` C60: Near observed clean gain -0.38% to +1.83%; CA_NATIVE -3.01% to -1.42% despite 9.5%–14.4% fewer decode peer bytes than BR.
- The C30 NVSwitch and P2P-disabled BR comparisons change sign across matched seeds, so these one-shot results do not establish a stable transport premium. P2P-disabled uses SHM on this same server; it is not a measurement on a physically non-NVLink server.
- The EP-window sum is a bounded diagnostic of rank completion/communication behavior. It can diverge from the clean TPOT ordering because instrumentation and non-EP work affect the separate run; no pure communication-versus-expert causal attribution is claimed.

Policy effects must be computed within each same-seed, same-cache, same-transport group using clean TPOT. C60/NVSwitch reference rows are in SHAREGPT_RESULTS.md. All other timing columns are unnormalized seconds from one separate instrumented 32-step decode. “First dispatch→last return” is a single contiguous interval across the whole decode; it includes intervening attention/controller/H2D work and is not EP-only. “EP layer windows” sums the 48 nonoverlapping earliest-dispatch-to-latest-return intervals per decode step. Compute and dispatch/return are exclusive scopes on each layer’s last-return rank; arrival/other is remaining window time, including rank arrival skew and non-EP work after the first rank starts. Current-stream spans can include host gaps, peer waits, and implicit H2D dependencies; they are not pure kernel/network service. Never subtract these diagnostic values from clean TPOT. One timing sample per policy does not establish statistical stability.
