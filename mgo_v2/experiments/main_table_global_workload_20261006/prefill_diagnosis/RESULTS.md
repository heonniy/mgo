# Remaining prefill TTFT diagnosis

Diagnostic only: one cold one-token run per cell after one-token warmup, Near/H0/full-pinned optimized prefill, R4/C30 on0/1/4/5. No default runtime change or primary-result replacement. All tokens match the prior optimized primary first token for the same requests. Cache, finite logits and no-compilation guards pass.

## Additive current-stream completion partition

Seconds. Critical rank is the rank with the latest first token in that cell. Each critical-rank column reconciles to global TTFT. Rank-mean columns sum to mean local TTFT; remaining completion skew is separately reported. Do not sum maxima chosen independently from each layer.

| Component | B16 rank mean | B16 critical | B64 rank mean | B64 critical |
|---|---:|---:|---:|---:|
| Router/Gate | 0.003869 | 0.003921 | 0.016267 | 0.016274 |
| Metadata counts | 0.024899 | 0.024875 | 0.013303 | 0.013827 |
| Metadata selected IDs | 0.013287 | 0.013731 | 0.012343 | 0.011866 |
| Metadata weights | 0.010566 | 0.010619 | 0.009516 | 0.009587 |
| Metadata Gate probability tail | 0.010232 | 0.010837 | 0.009374 | 0.009037 |
| D2H/materialization | 0.043459 | 0.045154 | 0.188528 | 0.184023 |
| Placement CPU | 0.159041 | 0.155165 | 1.115331 | 1.130619 |
| Layout CPU | 1.202698 | 1.173913 | 11.065628 | 11.410593 |
| Layout GPU tensor construction | 0.548090 | 0.540379 | 4.094870 | 4.163891 |
| H2D enqueue | 0.010806 | 0.010644 | 0.012835 | 0.012901 |
| Exposed H2D wait | 0.297090 | 0.285658 | 0.301548 | 0.306592 |
| Global barrier | 0.020581 | 0.032591 | 0.033803 | 0.012788 |
| Forward pack/A2A/completion | 0.115982 | 0.158382 | 0.711115 | 0.234454 |
| Expert execution | 1.883986 | 1.903598 | 1.926869 | 1.943663 |
| Return A2A/combine | 0.557191 | 0.538090 | 0.720778 | 0.712448 |
| Attention/dense/non-MoE | 0.072842 | 0.073026 | 0.482018 | 0.482084 |
| Other MoE/host gaps | 0.123556 | 0.118895 | 0.609406 | 0.678252 |
| Boundary residual | 0.000121 | 0.000074 | 0.000135 | 0.000193 |
| **Total** | 5.098295 | 5.099552 | 21.323666 | 21.333094 |

## Host CPU corroboration

Seconds, critical rank. CPU time is a separate view and must not be added to the above partition.

| Component | B16 own-thread CPU | B64 own-thread CPU |
|---|---:|---:|
| Placement CPU | 0.155314 | 1.129266 |
| Layout CPU | 1.174084 | 11.399613 |
| Layout GPU tensor construction | 0.540510 | 4.160092 |
| Expert execution | 1.903802 | 1.940909 |

## H2D service: non-additive

| Cell | Mean DMA service s/rank | Mean exposed wait s/rank | Global copies | Global GiB | Completion skew s |
|---|---:|---:|---:|---:|---:|
| B16/L256 | 0.285089 | 0.297090 | 6067 | 53.323242 | 0.001685 |
| B64/L512 | 0.286993 | 0.301548 | 6119 | 53.780273 | 0.012600 |

## Finding and next decision

CPU layout construction plus device-index materialization account for 73.01% of B64 critical-rank TTFT. Their large own-thread CPU times corroborate actual host work, not merely host descheduling or GPU wait. The largest B16-to-B64 increases in rank-mean seconds are: Layout CPU +9.863s, Layout GPU tensor construction +3.547s, Placement CPU +0.956s, Forward pack/A2A/completion +0.595s.

Prioritize the CPU token-index/layout representation path before grouped expert execution or H2D overlap. `plan_layout` performs token/expert sorting, per-expert scans and Python-list conversion on every layer. `pack_layouts` converts those lists back to NumPy, concatenates them and constructs device views. It was designed for outside-measurement immutable layouts but is called inside live prefill. Exact-mode `return_order` and per-expert `combine` are still built/packed even though rank-partial combine does not consume them. These are code-identified targets; sub-operation shares and repair gains have not been measured. No production repair is performed by this diagnostic.

H0 expert service and communication are secondary to this measured layout bottleneck. Expert intervals include host launches and gather/weight operations, not pure GEMM kernel activity. Communication completion includes peer arrival wait; do not interpret it as standalone link bandwidth. The metadata D2H reduction is already enabled. Global barrier/H2D waits are visible but are not the dominant remaining term.

## Limits and evidence

One diagnostic per cell, without randomization/repeats. This localizes the remaining optimized OURS prefill cost; it does not profile DeepSpeed or prove an exact cross-system causal delta. Additional event instrumentation can perturb execution. Current-stream CUDA spans include CPU gaps, allocator work and dependent-stream waiting; host wall/CPU are independent explanatory views. H2D copy spans are separate DMA-stream service and must not be summed again. Dense residual includes attention, other dense/model work and token selection. All48 layers/four ranks, source commit, raw intervals and copy records are preserved. Full1Hz resource logs remain in raw directories. No new work on GPUs2/3/6/7.
