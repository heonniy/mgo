# Qwen R4 grouped execution and placement scaling

Use Qwen3-30B, ShareGPT, four physical GPUs 0/1/4/5, C30, input 512 and
decode 64. Compare rank-local B8, B16 and B64. B8 is a fixed per-rank
eight-request subset of the frozen B16 manifests; B16 and B64 reuse their
existing frozen manifests and hashes. All executions use rank-private pinned
host experts, prefetch OFF, the same cache reset after warmup, compiled
prefill/decode layout and compiled dense decode routing weights.

Run `new_OURS` strict `hit_then_miss` grouping under BR, CA_NATIVE (reported
as CA) and LA_CA_NEAR (reported as Near) for every batch. Run two clean
uninstrumented repetitions per cell and preserve both values. Separately
profile one full decode per cell. Within each batch, compare policy effects
on TPOT, MAIN hit/miss count, H2D bytes and copy service, routed token-expert
rows, per-rank expert load, controller/index work, forward dispatch, grouped
expert execution, return/combine, attention/dense and unclassified runtime.
Use a rank with the largest expert-execution span as a representative
current-stream partition for each diagnostic cell, and retain the partition
for every rank. Collective boundaries align their total elapsed times, so
the overall slowest rank alone does not identify the cause. Sum mutually
exclusive segments to 100% of each separately instrumented TPOT.
Copy-stream service may overlap these spans and must not be added to 100%.
Report copy service and exposed wait separately. Collective intervals contain
peer arrival and host submission, so do not label them pure network wire
time. Show both absolute ms/token and growth from B8 to B16 to B64.
The second grouped wave is ordered behind the first and all miss-copy
completion. A GPU-stream-wait implementation with identical tokens/cache/H2D
was measured separately and regressed B64 TPOT, so the primary N schedule
keeps the faster host-side wait. Do not assume that moving a wait to the GPU
automatically shortens the end-to-end critical path.

Use the current main_OURS C++ Ready-First individual expert executor as the
causal baseline for the Near policy. At B64, compare default metadata/index
path with the compiled-dense candidate and the strict grouped schedule
separately. The grouped kernel may change BF16 rounding, generated tokens
and later live routing; report first divergence and demand/H2D differences.
Do not claim a same-trace scheduling gain if paths diverge. Keep raw prompts
and large diagnostics outside Git. Each timed job runs with the eight owned
background model loads stopped and restores them afterwards. Check GPU
resident processes, per-rank CPU affinity, available RAM and GPU headroom.
