# ShareGPT C60 policy-gap physical results

R4 on GPUs 0,1,4,5; C60 MAIN3678 plus eight reserved P2 slots; native expert execution, compiled metadata, full pinned CPU source, prefetch OFF. Each case has 128 input tokens, 32 decode forwards, and one clean primary per policy after warmup. Routes and teacher inputs are frozen across BR, CA_NATIVE, and Near.

| B/rank | Seed sample/order | Policy | Clean TPOT s/token | Paired diagnostic total s/token | EP span s/token | Expert compute s/token | Expert comm s/token | Peer GiB | H2D GiB |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20/7 | BR | 0.432621 | 0.481524 | 0.297756 | 0.187485 | 0.110271 | 1.026 | 204.346 |
| 8 | 20/7 | CA_NATIVE | 0.437821 | 0.492056 | 0.306581 | 0.190635 | 0.115946 | 0.878 | 205.910 |
| 8 | 20/7 | LA_CA_NEAR | 0.423135 | 0.492048 | 0.302899 | 0.190459 | 0.112440 | 0.996 | 203.871 |
| 8 | 14/5 | BR | 0.418036 | 0.479552 | 0.296258 | 0.179536 | 0.116722 | 1.024 | 160.497 |
| 8 | 14/5 | CA_NATIVE | 0.432362 | 0.480970 | 0.298671 | 0.180239 | 0.118432 | 0.894 | 162.343 |
| 8 | 14/5 | LA_CA_NEAR | 0.408085 | 0.459115 | 0.277386 | 0.176912 | 0.100473 | 0.990 | 159.917 |
| 16 | 22/3 | BR | 0.505627 | 0.561636 | 0.364611 | 0.231462 | 0.133149 | 2.050 | 370.222 |
| 16 | 22/3 | CA_NATIVE | 0.511198 | 0.573816 | 0.373395 | 0.233831 | 0.139564 | 1.804 | 371.259 |
| 16 | 22/3 | LA_CA_NEAR | 0.501507 | 0.560445 | 0.360793 | 0.239216 | 0.121577 | 1.966 | 370.090 |
| 16 | 16/3 | BR | 0.478414 | 0.533777 | 0.341265 | 0.223715 | 0.117550 | 2.048 | 320.757 |
| 16 | 16/3 | CA_NATIVE | 0.501038 | 0.553818 | 0.359466 | 0.228828 | 0.130638 | 1.817 | 323.209 |
| 16 | 16/3 | LA_CA_NEAR | 0.476265 | 0.537243 | 0.342499 | 0.227216 | 0.115283 | 1.976 | 320.581 |
| 64 | 13/0 | BR | 0.572153 | 0.637286 | 0.423875 | 0.289243 | 0.134631 | 8.215 | 558.510 |
| 64 | 13/0 | CA_NATIVE | 0.587160 | 0.650155 | 0.431554 | 0.287207 | 0.144347 | 7.406 | 559.354 |
| 64 | 13/0 | LA_CA_NEAR | 0.579188 | 0.636971 | 0.421445 | 0.283231 | 0.138214 | 8.009 | 558.976 |
| 64 | 20/5 | BR | 0.566874 | 0.630023 | 0.419380 | 0.283317 | 0.136064 | 8.219 | 543.771 |
| 64 | 20/5 | CA_NATIVE | 0.580988 | 0.644722 | 0.429313 | 0.280918 | 0.148395 | 7.438 | 545.177 |
| 64 | 20/5 | LA_CA_NEAR | 0.569150 | 0.628341 | 0.414817 | 0.280344 | 0.134473 | 8.010 | 544.263 |

Best observed clean TPOT gain versus the paired BR among the two measured physical seeds per batch (negative means slower):

| B/rank | CA_NATIVE seed / gain | Near seed / gain |
|---:|---|---|
| 8 | 20/7 / -1.20% | 14/5 / +2.38% |
| 16 | 22/3 / -1.10% | 22/3 / +0.81% |
| 64 | 20/5 / -2.49% | 20/5 / -0.40% |

Clean TPOT includes attention and is the performance result. Diagnostic total and EP span come from the same separate instrumented run. For each decode step, select the rank with the largest complete phase span. EP span is the exclusive sum of forward dispatch/finish, native expert execution, and return partial/combine scopes on that rank; controller, routing metadata, attention, and explicit H2D scopes are excluded. These current-stream spans can still contain host submission gaps, peer waits, and implicit H2D dependencies, so they are not pure NCCL/kernel service times. Instrumentation can slow the run; never subtract diagnostic EP from clean TPOT. One clean sample per policy does not establish repeatability.
