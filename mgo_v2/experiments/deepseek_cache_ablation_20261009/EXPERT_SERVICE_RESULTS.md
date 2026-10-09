# One DeepSeek expert: copy versus GEMM service

The guarded standalone `dca_expert_service_probe_v1` ran on physical GPU 0,
with owner model loads stopped and restored on GPUs 0/1/4/5. A DeepSeek
routed expert occupies **16.5 MiB BF16**: gate/up matrices of 1408×2048 and
a down matrix of 2048×1408. The probe used one pinned CPU expert source,
10 warmups and 50 CUDA-event samples per case.

| Operation | Median CUDA interval |
|---|---:|
| One pinned-host-to-GPU 16.5-MiB expert copy | 0.391 ms |
| Three expert GEMMs, 1 row | 0.048 ms |
| Three expert GEMMs, 3 rows | 0.046 ms |
| Three expert GEMMs, 5 rows | 0.045 ms |
| Three expert GEMMs, 8 rows | 0.045 ms |
| Three expert GEMMs, 17 rows | 0.046 ms |

The contiguous three-GEMM interval excludes input gather, activation,
routing-weight application, ready-wave scheduling, return combine and host
work outside the event interval. CUDA event intervals can contain launch gaps,
so these are not pure Nsight kernel-active times. The copy is about 8.7×
the isolated five-row three-GEMM interval, but this ratio is **not** a TPOT
prediction. The runtime transfers weights on a separate stream and executes
already-ready experts while copies continue. Many expert invocations, host
control work and cross-rank collectives determine each token's completion.

The full-model C20/C50 diagnostic measured logical H2D bytes and explicit
waits, not a per-copy kernel distribution. This standalone copy number is a
local service-time scale; it does not establish the exposed H2D critical path
under full-model contention. The matched-route D1 runs and sampled D3
profiling are needed for that. Aggregate samples are in `EXPERT_SERVICE.json`;
the supervisor receipt and raw probe result remain outside Git.
