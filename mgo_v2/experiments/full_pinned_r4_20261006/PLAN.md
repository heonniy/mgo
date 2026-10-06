# Rank-private Full Pinned Expert Store — R4 A/B

Goal: test the simplest full implementation suggested by the 897abf4 microbenchmark.

Each of the four ranks on physical GPUs 0,1,4,5 allocates its own complete pinned
copy of all Qwen3-30B-A3B experts:
- 6,144 experts/rank
- 9 MiB/expert
- 54 GiB pinned host memory/rank
- 216 GiB total pinned host memory for R4

This is intentionally memory-inefficient but simple. CPU memory is treated as
available for this experiment.

## Runtime change

STAGED:
file-backed expert -> two pinned staging buffers -> GPU cache.

FULL_PINNED:
rank-private contiguous pinned expert -> existing PriorityH2DScheduler ->
dedicated H2D stream -> GPU cache.

FULL_PINNED keeps the scheduler's priority queue, prefetch promotion, slot
hazard events and completion tracking. Only the CPU staging memcpy is removed.
This avoids the scheduler-bypass confound in the earlier 288-MiB microbenchmark.

## Physical A/B

Reuse the established optimized physical workload:
- R4 physical GPUs 0,1,4,5 only
- C30
- local batch 128
- prefill from the existing frozen requests
- decode 64
- BR placement, identical frozen routes/fetches
- V3 optimized prefetch-overlap runtime
- P=2, T2
- BF16 partials
- unique combine
- async metadata inputs
- H1b graph expert executor
- no diagnostic post-expert barrier

Compare:
1. STAGED
2. FULL_PINNED

One correctness/counter pass per mode, then two counterbalanced clean timing
repeats per mode. Full pinned allocation/copy is outside timing.

Report TTFT (=E2E-decode wall), TPOT, decode wall, E2E, peak HBM, max RSS,
scheduler copies/bytes and full pinned initialization time.

This is an implementation A/B before the main baseline table, not a final
paper number.

## Safety
- Never touch GPUs 2,3,6,7.
- Stop/restore only this project's idle model loads on 0,1,4,5.
- Require >=384 GiB host MemAvailable before launch.
- Abort if host MemAvailable falls below 96 GiB.
- 30-minute hard runtime bound.
