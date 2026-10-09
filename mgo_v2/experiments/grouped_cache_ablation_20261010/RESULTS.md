# R4 new_OURS cache sweep results

All four physical runs passed on 2026-10-10. The workload is the frozen
Qwen3-30B ShareGPT R4/GPU-local B16/input512/output64 workload on physical
GPUs 0/1/4/5. `new_OURS` means native C++ prefill, compiled dense routing,
Near placement, prefetch OFF, and strict hit-grouped-GEMM followed by a
single grouped-GEMM for all misses after their H2D transfers complete.
Each job includes one warmup and two unfiltered, cache-reset measured batches.

TTFT and TPOT are seconds and seconds per output token, respectively. TPS is
**global** output tokens per second, `64 requests × 64 tokens / E2E seconds`.
Entries below are two-sample means with the full range in brackets; both
individual samples appear in the raw job receipts.

| Cache | TTFT, s | TPOT, s/token | TPS, token/s | TPOT repeat difference |
|---:|---:|---:|---:|---:|
| C20 | 2.706 [1.752, 3.660] | 0.3499 [0.3461, 0.3537] | 165.63 [160.87, 170.40] | 2.20% |
| C30 | 2.709 [1.758, 3.661] | 0.3286 [0.3210, 0.3362] | 175.03 [171.51, 178.55] | 4.64% |
| C40 | 2.696 [1.757, 3.636] | 0.3025 [0.3004, 0.3046] | 188.53 [181.53, 195.52] | 1.39% |
| C50 | 2.704 [1.756, 3.651] | 0.2771 [0.2749, 0.2793] | 203.73 [192.75, 214.71] | 1.59% |

The measured TPOT ranges do not overlap between adjacent cache sizes. C20
to C50 reduces mean TPOT by 20.8% and increases mean global TPS by 23.0%.
TTFT shows the same systematic first/second gap in **every** cache cell:
first measurement 3.636–3.661 s, second 1.752–1.758 s. Thus its mean is a
description of the two observed timings, not a stable TTFT estimate. TPS
inherits some of this TTFT variation. The measured runs show no recompilation
in their target phases; the precise source of the first-target TTFT gap is
not established by this sweep. No samples were discarded or selected.

The other R4 baselines have already been measured on the same frozen target
requests. This table reports their validated median values from
[`BASELINE_SWEEP_RESULTS.json`](../qwen_cache_ablation_20261009/BASELINE_SWEEP_RESULTS.json).
TPS is recomputed as the median of `4096 / E2E` for their individual samples.
These are earlier runs with each system's audited implementation, so this is
an observational comparison across run dates, not a paired timing experiment.

| Cache | Baseline | TTFT, s | TPOT, s/token | TPS, token/s | Targets |
|---:|---|---:|---:|---:|---:|
| C20 | MoE-Infinity | 5.207 | 3.143 | 20.16 | 3 |
| C20 | DeepSpeed | 5.886 | 4.045 | 15.71 | 2 |
| C20 | llama.cpp, synchronous balanced | 161.936 | 0.525 | 21.00 | 2 |
| C30 | MoE-Infinity | 5.192 | 3.227 | 19.64 | 3 |
| C30 | DeepSpeed | 5.571 | 4.122 | 15.40 | 3 |
| C30 | llama.cpp, synchronous balanced | 145.070 | 0.481 | 23.36 | 3 |
| C40 | MoE-Infinity | 5.548 | 3.204 | 19.75 | 3 |
| C40 | DeepSpeed | 5.951 | 4.112 | 15.46 | 2 |
| C40 | llama.cpp, synchronous balanced | 129.525 | 0.436 | 26.10 | 2 |
| C50 | MoE-Infinity | 5.562 | 3.244 | 19.51 | 2 |
| C50 | DeepSpeed | 5.975 | 4.147 | 15.33 | 2 |
| C50 | llama.cpp, synchronous balanced | 103.026 | 0.343 | 32.86 | 2 |

The older selected `main_OURS` two-repeat R4 results are separately reported
in [`FULL_CACHE_COMPARISON.md`](../qwen_cache_ablation_20261009/FULL_CACHE_COMPARISON.md).
Its decode path is the native Ready-First individual expert executor, whereas
this experiment changes decode to grouped expert execution. Do not relabel
the older rows as `new_OURS`.

Raw receipts are in `/home/hwlee/mgo-results/grouped_cache_ablation_20261010/jobs/`.
All four guarded jobs returned `PASS`, verified empty matching cache state at
each target start, recorded prefetch OFF and no target recompilation on all
four ranks, kept peak allocated HBM at or below 12.19 GB per rank, and
restored all eight owned model loads after each job. No measured worker
processes remained before restoration. GPU memory subsequently returned to
its normal eight-load level (37,183 MiB per GPU).
