# Frozen-route grouped placement search

Compare `new_OURS` strict hit-first/miss-second grouped decode at R4 on
physical GPUs 0/1/4/5 under three placements: balanced random (BR),
`LA_CA_NEAR` (Near), and a static expert owner `expert_id % 4` (Static).
Keep native C++ prefill, Qwen-only compiled dense decode routing,
rank-private pinned source, prefetch OFF, BF16, and the same C30/C60
MAIN-slot budget for all policies. Static retains local expert eviction
within its fixed owner rank; it does not replicate all experts in HBM.

Use the eight existing frozen ShareGPT seed cells in
`/home/hwlee/mgo-results/policy_gap_c60_20261008/FOLLOWUP_WORKLOADS.json`:
C30/C60 × B8 (two seeds), B16 (one), B64 (one), input128, 32 decode
forwards. This is a bounded seed search, not a claim of global optimality.
For each cell, capture routing and teacher tokens once under grouped BR,
then replay identical route tensors and teacher inputs for all three
policies. Run two unfiltered clean primaries per policy in the
counterbalanced order BR/Near/Static/Static/Near/BR. Reset the expert cache
before each run and keep raw route hashes, policy output agreement, H2D,
peer traffic, and rank-local receipts outside Git. A tiny two-decode smoke
must pass before the full queue. After screening, repeat the best positive
Near-versus-BR candidate in a fresh guarded job to check stability.

Use one guarded GPU job at a time, retain host/HBM checks, do not kill
foreign processes, and restore the owned model loads. Report TPOT for
BR/Near/Static in three adjacent columns with full ranges and the exact
seed, cache and batch. TTFT and global TPS are secondary. A positive gain
is `(BR TPOT - Near TPOT) / BR TPOT`; do not select individual favorable
repeats or label a gain reliable when the paired ranges overlap materially.
Static is a different placement constraint: its unequal rank fetch counts
are expected, while its per-rank slot capacity and fixed-owner invariant
must pass.

If all eight NVSwitch cells show no positive Near mean gain, extend the
bounded search to the four C30 seeds under the established same-host env2
transport (`NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`). Keep every other
condition and the two-run order unchanged. Label this transport separately;
do not pool its timings with NVSwitch.
