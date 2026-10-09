# One Qwen expert: H2D versus GEMM service

The guarded, standalone probe `qca_expert_service_probe_v2` ran on physical
GPU 0 with the four owner model loads stopped and then restored. It allocated
one Qwen BF16 expert of exactly 9 MiB (gate/up/down matrices with shapes
768×2048, 768×2048, and 2048×768). Its source was pinned CPU memory. The
probe used 50 CUDA-event samples after 10 warmups per case; it did not load
the model or run dispatch, communication, the controller, or attention.

| Operation | Median CUDA interval |
|---|---:|
| One pinned-host-to-GPU 9-MiB expert copy | 0.214 ms |
| Three Qwen expert GEMMs, 1 row | 0.047 ms |
| Three Qwen expert GEMMs, 3 rows | 0.046 ms |
| Three Qwen expert GEMMs, 5 rows | 0.045 ms |
| Three Qwen expert GEMMs, 8 rows | 0.045 ms |
| Three Qwen expert GEMMs, 17 rows | 0.046 ms |

The **three-GEMM probe excludes** input gather, SiLU/multiply, routing-weight
application, ready-wave scheduling, slot events, output combine and host
launch overhead outside the CUDA interval. Each individual GEMM measured
about 0.020–0.022 ms, but adding three separate event intervals double-counts
submission gaps; the contiguous three-GEMM interval is the relevant figure.
CUDA events include gaps between kernel submissions, so these figures are
service intervals, not an Nsight kernel-active-time decomposition. The older
native-executor resident microbenchmark measured **0.166–0.169 ms per expert**
for the complete gather/three-GEMM/activation/weighting path, including host
loop and synchronization amortized over 32 experts. That is a different
workload, and it is not pure GEMM time.

The full-model C20/C50 diagnostic independently measured median 9-MiB copy
service of **0.182–0.228 ms across ranks**, consistent in scale with the
standalone 0.214 ms. A fetch can therefore cost several times the isolated
three-GEMM interval for a small expert group, but one cannot multiply this
ratio by demand misses to predict TPOT: copy-stream work overlaps expert
execution, many experts execute per layer, and the slowest rank and collective
arrival determine the token critical path.

Raw samples and the supervisor's resource/restore receipt remain in
`/home/hwlee/mgo-results/headline_r4_20261007/qca_expert_service_probe_v2/`.
