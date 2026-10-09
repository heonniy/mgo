# R4 main_OURS decode hot-path optimization

## Scope and result

Qwen3-30B, ShareGPT, R4 on physical GPUs 0/1/4/5, cache C30, local B16,
input 512, output 64, Near placement, prefetch OFF and the selected C++
Ready-First expert executor. All cells use the same frozen prompts and cold
cache reset after warmup. This study does not change the opt-in B/C grouped
expert executor or the selected main_OURS expert executor.

| Version | Decode change | TPOT repeat 1 / 2 (s/token) | Mean TPOT (s/token) | Mean E2E (s) |
|---|---|---:|---:|---:|
| A v1 | Original | 0.500185 / 0.499571 | 0.499878 | 34.183932 |
| A v3 | Compiled gate-history update | 0.499451 / 0.500755 | 0.500103 | 34.212731 |
| A v4 | Above + bulk tensor-view split for layout | 0.494048 / 0.495489 | 0.494769 | 33.873676 |
| A v5 | Above + opt-in fused decode routing weights | 0.478979 / 0.477177 | **0.478078** | 32.810120 |

V5 improves mean TPOT by **4.36%** versus v1 and **3.37%** versus v4.
The v5 repeat spread is 0.38% relative to their mean. TTFT means are
2.691617 seconds in v1 and 2.691223 seconds in v5; the first target's TTFT
is much slower in both versions, so this change is a decode result, not a
prefill claim. All 4 ranks × 2 full repeats matched v1 and v4 exactly in
generated tokens, final cache-state hash and H2D bytes. Each guarded job
restored eight owned model loads. The candidate also passed an R4 smoke and a
single-GPU exact BF16 differential test at batches 1, 16 and 64, including
colliding substitution destinations.

Primary job receipts are under
`/home/hwlee/mgo-results/expert_grouping_ablation_20261009/jobs/r4_a_full_v{1,3,4,5}`.
The two changes in v4 were committed as `288e2af`; the opt-in candidate was
committed as `3efe1df`. The v4/v5 primary jobs were started with dirty
worktrees and then committed unchanged in their execution paths; their
`source_commit` receipt alone identifies the preceding HEAD, not the entire
executed tree. The v5 diagnostic was run after commit `3efe1df`.

## Where time was removed

These figures are from separate instrumented full-decode runs and are the
mean of per-rank ms/token, not primary TPOT components. Instrumentation raises
the absolute time, and collective intervals include peer waits.

| Phase | Original A v1 | Gate/layout A v4 | Fused dense A v5 |
|---|---:|---:|---:|
| Gate-history update | 9.80 | 6.17 | 6.17 |
| GPU layout tensor materialization | 12.82 | 10.20 | 10.38 |
| Unclassified MoE runtime | 49.99 | 48.79 | 29.47 |
| Fused dense routing kernel | — | — | 3.76 |

The original gate-history update used a Python row loop and NumPy vector
operations. The compiled update preserves the reference FP64 add-then-evict
order and the canonical deque; randomized partial/full-window tests were
bitwise exact. The original layout pack made two Python tensor slices per
expert group; a single `torch.split` now creates those views while retaining
the same packed index buffer. The 100-case CPU differential layout test passed.

In v4, the largest unclassified interval, from layout materialization to
demand-H2D submission, averaged 29.23 ms/token. This interval creates a
128-column dense routing-weight matrix with zero, indexed scatter-add and
conversion on every layer. V5's opt-in Triton kernel builds those weights in
one pass, reusing a decode buffer. V4's unclassified MoE time was 48.79
ms/token; v5's unclassified time plus its explicit dense-kernel phase was
33.23 ms/token, a 15.56 ms/token reduction. Primary TPOT fell by 16.69
ms/token from v4 to v5. This close agreement supports, but does not alone
prove, the dense-weight chain as the main removed cost.

## Dispatch, return and remaining work

The packet's metadata all-gather and token dispatch/return intervals remain
large, but their CUDA-event spans include peer arrival, host submission gaps
and local packet/partial operations. The metadata all-gather changed from
26.83 ms/token rank mean in v1 to 29.82 in v4; this is not an optimization
target justified by the local CPU edits. V5's return all-to-all interval
varied from 26.5 to 57.1 ms/token across ranks in the diagnostic. There is no
evidence here that rewriting NCCL or reducing payload bytes alone would
remove that elapsed time. A next controlled study should separate local
partial accumulation, rank arrival wait, collective completion and final
scatter; it must retain token/cache parity before promoting any return
fusion. The opt-in dense kernel is Qwen top-8/128-expert specific and has
not been promoted to the shared main_OURS default or tested on DeepSeek.

The implementation direction matches the fused token permutation and
dispatch paths documented by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md),
[DeepEP](https://github.com/deepseek-ai/DeepEP) and
[vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/).
Those systems are architectural references, not drop-in replacements for
dynamic offload and placement. Current DeepEP installation requirements in
its README include CUDA 13.1+, while this experiment environment uses
PyTorch 2.11 with CUDA 12.8.
