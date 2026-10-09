# Qwen cache-capacity comparison after timing stabilization

The original chronological four-job sweep showed a 25% C20→C50 TPOT gap,
but independent jobs later varied by more than that while producing identical
outputs, H2D bytes and cache hashes at a fixed capacity. The table below is
the **post-cleanup recheck**: one guarded job per capacity, each with a
separate warmup and two unfiltered cold-cache target repeats. All use the
same frozen ShareGPT R4 local-B16/input512/output64 requests, GPUs 0/1/4/5,
Near/native C++/compiled main_OURS, full pinned source and prefetch OFF.

| Cache | TPOT, ms/token, both repeats | TPOT mean, ms/token | Relative to C20 | Total batch H2D, GiB, four ranks |
|---:|---:|---:|---:|---:|
| C20 | 502.488, 504.161 | 503.324 | baseline | 1,913.010 |
| C30 | 493.509, 496.027 | 494.768 | 1.70% faster | 1,578.313 |
| C40 | 492.758, 493.217 | 492.987 | 2.05% faster | 1,246.421 |
| C50 | 488.052, 489.795 | 488.924 | 2.86% faster | 959.001 |

Each recheck repeat exactly matched its corresponding original-capacity run
in request IDs, all output tokens, H2D bytes, final MAIN ownership/cache
hashes, finite logits, and no recompilation. Different capacities produced
different later tokens for some requests, so this is an empirical live
generation comparison rather than a same-route causal copy-removal test.

Logical decode MAIN hit rate rose from 24.83% (C20) to 63.49% (C50), and
decode H2D fell from 1,859.5 to 905.5 GiB. The original sweep's 25% TPOT
gap should not be used as the cache benefit; it coincided with large
cross-job timing drift. This stable recheck supports a **small, roughly 3%**
TPOT improvement across C20–C50 on this workload, with diminishing returns
after C30. Because copy-stream work overlaps with expert execution, byte
reduction alone is not an estimate of exposed latency. A targeted overlap
ablation follows; the stale Nsight agents found on the host are a possible
noise source, not a proven sole cause of the original drift.

The paired TTFT and E2E samples, exact source commits, and per-job validation
are in `TIMING_RECHECK.json`. TTFT still has an elevated first target in each
job, so its two-sample mean should not be treated as a stable TTFT estimate.
