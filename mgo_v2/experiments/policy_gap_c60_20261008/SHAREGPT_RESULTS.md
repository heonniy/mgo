# ShareGPT C60 policy-gap physical results

R4 on GPUs 0,1,4,5; C60 MAIN3678 plus eight reserved P2 slots; native expert execution, compiled metadata, full pinned CPU source, prefetch OFF. Each case has 128 input tokens, 32 decode forwards, and one clean primary per policy after warmup. Routes and teacher inputs are frozen across BR, CA_NATIVE, and Near.

| B/rank | Seed sample/order | Policy | Clean TPOT ms/token | Diagnostic full decode s | First dispatch→last return s | EP layer windows s | Expert compute s | Dispatch/return s | Arrival/other s | Peer GiB | H2D GiB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20/7 | BR | 432.621 | 15.409 | 15.404 | 10.037 | 6.835 | 2.807 | 0.396 | 1.026 | 204.346 |
| 8 | 20/7 | CA_NATIVE | 437.821 | 15.746 | 15.741 | 10.286 | 7.078 | 2.804 | 0.404 | 0.878 | 205.910 |
| 8 | 20/7 | LA_CA_NEAR | 423.135 | 15.746 | 15.741 | 10.207 | 6.882 | 2.895 | 0.430 | 0.996 | 203.871 |
| 8 | 14/5 | BR | 418.036 | 15.346 | 15.341 | 9.975 | 6.773 | 2.766 | 0.436 | 1.024 | 160.497 |
| 8 | 14/5 | CA_NATIVE | 432.362 | 15.391 | 15.386 | 9.972 | 7.071 | 2.570 | 0.331 | 0.894 | 162.343 |
| 8 | 14/5 | LA_CA_NEAR | 408.085 | 14.692 | 14.687 | 9.311 | 6.528 | 2.486 | 0.297 | 0.990 | 159.917 |
| 16 | 22/3 | BR | 505.627 | 17.972 | 17.967 | 12.156 | 8.565 | 3.194 | 0.397 | 2.050 | 370.222 |
| 16 | 22/3 | CA_NATIVE | 511.198 | 18.362 | 18.357 | 12.479 | 8.599 | 3.420 | 0.460 | 1.804 | 371.259 |
| 16 | 22/3 | LA_CA_NEAR | 501.507 | 17.934 | 17.929 | 12.062 | 8.448 | 3.217 | 0.397 | 1.966 | 370.090 |
| 16 | 16/3 | BR | 478.414 | 17.081 | 17.076 | 11.430 | 7.862 | 3.165 | 0.403 | 2.048 | 320.757 |
| 16 | 16/3 | CA_NATIVE | 501.038 | 17.722 | 17.717 | 11.963 | 8.549 | 3.018 | 0.397 | 1.817 | 323.209 |
| 16 | 16/3 | LA_CA_NEAR | 476.265 | 17.192 | 17.187 | 11.392 | 8.038 | 3.028 | 0.325 | 1.976 | 320.581 |
| 64 | 13/0 | BR | 572.153 | 20.393 | 20.387 | 14.170 | 10.238 | 3.413 | 0.520 | 8.215 | 558.510 |
| 64 | 13/0 | CA_NATIVE | 587.160 | 20.805 | 20.799 | 14.457 | 10.461 | 3.450 | 0.546 | 7.406 | 559.354 |
| 64 | 13/0 | LA_CA_NEAR | 579.188 | 20.383 | 20.378 | 14.423 | 9.790 | 3.742 | 0.891 | 8.009 | 558.976 |
| 64 | 20/5 | BR | 566.874 | 20.161 | 20.155 | 14.045 | 10.110 | 3.372 | 0.562 | 8.219 | 543.771 |
| 64 | 20/5 | CA_NATIVE | 580.988 | 20.631 | 20.626 | 14.298 | 10.300 | 3.423 | 0.575 | 7.438 | 545.177 |
| 64 | 20/5 | LA_CA_NEAR | 569.150 | 20.107 | 20.101 | 13.844 | 9.951 | 3.403 | 0.490 | 8.010 | 544.263 |

Best observed clean TPOT gain versus the paired BR among the two measured physical seeds per batch (negative means slower):

| B/rank | CA_NATIVE seed / gain | Near seed / gain |
|---:|---|---|
| 8 | 20/7 / -1.20% | 14/5 / +2.38% |
| 16 | 22/3 / -1.10% | 22/3 / +0.81% |
| 64 | 20/5 / -2.49% | 20/5 / -0.40% |

Clean TPOT includes attention and is the performance result. All other timing columns are unnormalized seconds from one separate instrumented 32-step decode. “First dispatch→last return” is a single contiguous elapsed interval across the whole decode; it necessarily includes intervening attention, controller, and H2D work, so it is not an EP-only duration. “EP layer windows” instead sums the 48 nonoverlapping earliest-dispatch-to-latest-return intervals per decode step. Compute and dispatch/return are exclusive scopes on each layer’s last-return rank; arrival/other is the remaining window time. Current-stream scopes can include host gaps, peer waits, and implicit H2D dependencies, so these are not pure NCCL/kernel service times. Instrumentation can slow the run; never subtract these diagnostic values from clean TPOT. One clean sample per policy does not establish repeatability.
