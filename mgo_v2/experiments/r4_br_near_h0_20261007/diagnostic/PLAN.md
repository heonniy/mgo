# Owner-requested one-cell BR versus Near attribution

Scope: B64/input512/C30/R4 GPUs0,1,4,5, decode32, BR and LA_CA_NEAR.
Owner narrowed eight diagnostics to one workload, then clarified that both
policies are required to explain the BR-to-Near TPOT difference.
Reuse completed primary frozen inputs, seed28, proofs and H0/full-pinned V3
P2/T2. One correctness warmup per policy followed by one diagnostic per policy.
No new routes, seeds, graph executor, strict phase barriers or extra repetitions.

CUDA events record decode MoE, compute, H2D dependency waits, metadata,
forward and return completion spans on the current stream. Existing scheduler
profile events record actual copy-stream service spans. No additional execution
synchronization is introduced. GPU spans include host dispatch gaps; collective
completion spans include peer arrival waits. They are not pure kernel or wire
latency. Controller CPU times and per-layer rank expert rows are separate.
Compute minus H2D wait is expert execution service, including host launch gaps.

Report clean primary TPOT separately from instrumented TPOT. Do not add DMA
service to overlapping compute/communication; distinguish exposed H2D dependency
wait. Rank imbalance is an indicator, not an independently additive TPOT cost.
One paired diagnostic supports attribution observations, not causal proof or
repeat stability. Preserve all four ranks and per-event records as compressed
JSON; validate CPU state, roles, counters, copy bounds, tokens and no recompiles.
384 GiB initial host headroom, abort below96 GiB, 30-minute bound. Restore owned
model load only on0,1,4,5 after completion/failure; never touch2,3,6,7.
