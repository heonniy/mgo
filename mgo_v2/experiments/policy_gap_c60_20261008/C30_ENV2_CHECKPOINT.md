# C30/env2 checkpoint

ShareGPT, R4 GPUs 0/1/4/5, input 128, 32 decode forwards, prefetch OFF. Env2 sets `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1`; an INFO smoke selected SHM. These four selected seed cases each have one clean primary and a separate instrumented MoE-block diagnostic. Values are seconds/token.

| B/rank | Seed case | Policy | Clean TPOT | MoE-block TPOT (diagnostic) |
|---:|---:|---|---:|---:|
| 8 | 0 | BR | 0.451557 | 0.465817 |
| 8 | 0 | CA_NATIVE | 0.458144 | 0.473057 |
| 8 | 0 | LA_CA_NEAR | 0.444598 | 0.448744 |
| 8 | 1 | BR | 0.436450 | 0.451503 |
| 8 | 1 | CA_NATIVE | 0.443018 | 0.458694 |
| 8 | 1 | LA_CA_NEAR | 0.425118 | 0.438044 |
| 16 | 0 | BR | 0.523418 | 0.534965 |
| 16 | 0 | CA_NATIVE | 0.529840 | 0.545386 |
| 16 | 0 | LA_CA_NEAR | 0.512613 | 0.522659 |
| 64 | 1 | BR | 0.594726 | 0.610510 |
| 64 | 1 | CA_NATIVE | 0.604834 | 0.624176 |
| 64 | 1 | LA_CA_NEAR | 0.585696 | 0.602023 |

MoE-block time excludes attention but includes host gaps and rank waits. Instrumentation changes timing; do not subtract it from clean TPOT. One sample does not establish repeatability.
