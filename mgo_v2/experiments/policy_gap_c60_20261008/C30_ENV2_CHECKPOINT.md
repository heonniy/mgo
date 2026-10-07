# C30/env2 checkpoint

ShareGPT, R4 GPUs 0/1/4/5, input 128, 32 decode forwards, prefetch OFF. Env2 sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`; an INFO smoke selected SHM. Values are seconds/token.

| B/rank | Seed case | Policy | Clean full TPOT | Paired diagnostic full | EP span | Expert compute | Expert comm |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 0 | BR | 0.451557 | 0.500436 | 0.319692 | 0.199904 | 0.119788 |
| 8 | 0 | CA_NATIVE | 0.458144 | 0.507289 | 0.321976 | 0.196579 | 0.125398 |
| 8 | 0 | LA_CA_NEAR | 0.444598 | 0.490677 | 0.306857 | 0.198979 | 0.107878 |
| 8 | 1 | BR | 0.436450 | 0.483359 | 0.305058 | 0.184127 | 0.120931 |
| 8 | 1 | CA_NATIVE | 0.443018 | 0.490300 | 0.307596 | 0.183777 | 0.123819 |
| 8 | 1 | LA_CA_NEAR | 0.425118 | 0.474113 | 0.293436 | 0.184888 | 0.108548 |
| 16 | 0 | BR | 0.523418 | 0.569269 | 0.374831 | 0.249329 | 0.125502 |
| 16 | 0 | CA_NATIVE | 0.529840 | 0.578340 | 0.376623 | 0.247135 | 0.129489 |
| 16 | 0 | LA_CA_NEAR | 0.512613 | 0.561630 | 0.363982 | 0.247102 | 0.116880 |
| 64 | 1 | BR | 0.594726 | 0.640577 | 0.433218 | 0.301173 | 0.132045 |
| 64 | 1 | CA_NATIVE | 0.604834 | 0.654264 | 0.436814 | 0.298617 | 0.138197 |
| 64 | 1 | LA_CA_NEAR | 0.585696 | 0.633487 | 0.422382 | 0.302163 | 0.120219 |

Clean full TPOT includes attention. Diagnostic full and EP come from one separately instrumented run. EP is the sum of dispatch/finish, expert execution, and return partial/combine spans on the slowest complete-phase rank per step. Controller, routing metadata, attention, and explicit H2D scopes are excluded. Current-stream spans can still contain host gaps, peer waiting, and implicit H2D dependencies; they are not pure kernel/network service. Do not subtract EP from clean TPOT. One sample per policy does not establish repeatability.
