# C30/env2 checkpoint

ShareGPT, R4 GPUs 0/1/4/5, input 128, 32 decode forwards, prefetch OFF. Env2 sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`; an INFO smoke selected SHM. Clean TPOT is ms/token. Other timing columns are seconds over all 32 decode steps from one separate diagnostic run.

| B/rank | Seed case | Policy | Clean TPOT ms/token | Diagnostic full decode s | First dispatch→last return s | EP layer windows s | Expert compute s | Dispatch/return s | Arrival/other s |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | BR | 451.557 | 16.014 | 16.009 | 10.749 | 6.956 | 3.333 | 0.460 |
| 8 | 0 | CA_NATIVE | 458.144 | 16.233 | 16.228 | 10.800 | 7.251 | 3.128 | 0.422 |
| 8 | 0 | LA_CA_NEAR | 444.598 | 15.702 | 15.697 | 10.294 | 6.839 | 3.085 | 0.370 |
| 8 | 1 | BR | 436.450 | 15.467 | 15.458 | 10.291 | 6.703 | 3.125 | 0.463 |
| 8 | 1 | CA_NATIVE | 443.018 | 15.690 | 15.684 | 10.428 | 6.408 | 3.515 | 0.505 |
| 8 | 1 | LA_CA_NEAR | 425.118 | 15.172 | 15.166 | 9.873 | 6.368 | 3.099 | 0.406 |
| 16 | 0 | BR | 523.418 | 18.217 | 18.211 | 12.514 | 8.868 | 3.245 | 0.401 |
| 16 | 0 | CA_NATIVE | 529.840 | 18.507 | 18.501 | 12.578 | 8.860 | 3.289 | 0.429 |
| 16 | 0 | LA_CA_NEAR | 512.613 | 17.972 | 17.967 | 12.203 | 8.482 | 3.259 | 0.461 |
| 64 | 1 | BR | 594.726 | 20.498 | 20.493 | 14.380 | 10.497 | 3.392 | 0.490 |
| 64 | 1 | CA_NATIVE | 604.834 | 20.936 | 20.931 | 14.575 | 10.583 | 3.421 | 0.570 |
| 64 | 1 | LA_CA_NEAR | 585.696 | 20.272 | 20.266 | 14.064 | 10.174 | 3.418 | 0.472 |

The single first-dispatch→last-return interval spans all 32 steps and includes attention/controller/H2D between EP windows; it is not EP-only. EP layer windows sum 48 nonoverlapping layer-local first-dispatch→last-return intervals per step. Compute and dispatch/return are exclusive scopes on each layer’s last-return rank. Arrival/other is remaining window time. Current-stream scopes can contain host gaps, peer waits, and implicit H2D dependencies; they are not pure kernel/network service. Do not subtract diagnostic times from clean TPOT. One clean sample per policy does not establish repeatability.
