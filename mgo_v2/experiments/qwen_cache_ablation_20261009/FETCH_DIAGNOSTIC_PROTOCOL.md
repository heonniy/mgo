# Qwen cache/fetch diagnostic after the primary sweep

Run C20 and C50 diagnostic jobs **after** all four clean main_OURS timings.
Use the same frozen ShareGPT target, same R4 GPUs 0/1/4/5, and the selected
native/compiled Near runtime with prefetch OFF. Each guarded job has a warmup,
one clean target, and a separate post-generation replay with phase events.
The replay must reproduce the target's complete output tokens and final cache
state; it is excluded from primary TPOT.

For each rank, report decode distinct-expert MAIN hits, demand misses, hit
rate, and expert H2D bytes. Also report the median/p95 CUDA copy-service time
per 9-MiB expert, total copies, explicit required-H2D wait on the main
stream, and the current-stream route, dispatch, expert and return spans. These
overlap and do not form an additive TPOT breakdown. Check target tokens/H2D
bytes against all three clean primary repeats; report cross-capacity token
differences rather than assuming fixed routing. A lower miss count with little
TPOT change would show cache effectiveness but **would not** prove fetch is
irrelevant: DMA can overlap and can indirectly contend with other work.

If the result remains ambiguous, diagnose fixed-continuation and frozen-route
execution separately before claiming a causal bottleneck. Do not change the
main_OURS implementation or replace the clean primary timings with profiled
timings.
