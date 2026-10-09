# D0: DeepSeek cache-plateau audit and replay contract

The prior DeepSeek-V2-Lite-Chat ShareGPT R4 experiment used local B16,
input512/output64, GPUs 0/1/4/5, Near/native execution, and prefetch OFF.
The C20/C50 primary median TPOT was **287.904/281.710 ms/token**, a
6.194 ms/token (2.15%) observed reduction. The separate diagnostics had
exact per-capacity output/H2D parity with their primary runs, but only 25/64
complete output sequences matched *between* capacities. Those runs therefore
do not supply a same-route causal comparison.

The committed diagnostic aggregate recomputes as follows. There are
26 routed layers × 63 decode intervals = 1,638 layer-intervals per rank;
the measured first generated token is prefill and is excluded here.

| Decode metric | C20 | C50 |
|---|---:|---:|
| H2D, four-rank sum | 1,397.489 GiB | 878.448 GiB |
| Executed expert groups, four-rank sum | 102,818 | 102,870 |
| Groups/rank/layer-interval | 15.693 | 15.701 |
| Native ready waves, four-rank sum | 23,908 | 13,423 |
| Waves/rank/layer-interval | 3.649 | 2.049 |
| Largest rank-mean explicit CUDA H2D wait | 0.327 ms/token | 0.037 ms/token |

Per-rank C20 expert-group counts were 26,270/25,872/25,530/25,146 and wave
counts were 6,845/6,493/5,152/5,418. At C50 they were
26,297/25,922/25,537/25,114 groups and 3,273/3,593/3,275/3,282 waves.
The wave reduction is a plausible mediator of the small TPOT gain but not
yet a controlled attribution: routes and cache placement differ. Prior
phase spans include host submission and peer waiting and cannot be summed as
separate service times. The earlier diagnostic raised TPOT by 3.2–3.8%.

## Replay manifests and fail-closed checks

Keep full manifests outside Git and commit their SHA-256 references only.
The common header must contain model revision, tokenizer/config digest,
request IDs and prompt-token hash, physical rank mapping, source commit,
workload manifest hash, BF16/determinism settings, and cache-capacity budget.

1. **Fixed continuation:** Store the reference next-token ID for every one of
   the 64 requests and 64 generated positions, in global rank/request order.
   During replay, execute the model normally and record its predicted argmax,
   but feed the reference ID as the next step's input. Hash each rank's actual
   input-ID tensor at each step. The C20/C50 hashes must match at every step;
   an argmax mismatch is reported separately and never replaced silently.
2. **Router capture:** For every `(step, routed layer, rank)`, store local
   selected expert IDs `[16, topk]`, BF16 routing weights `[16, topk]`, and
   FP32 full router probabilities `[16, 64]`, with dtype/shape and a SHA-256
   for each. Capture before the runtime's global metadata gather. Also retain
   resulting global selected/demand hashes, global gate-history score hash,
   owner assignments, expert groups/rows, dispatch/return counts and H2D.
3. **Frozen-router replay:** Inject all three captured router tensors at the
   patched MoE boundary, before `DeepSeekRuntime.execute`. Retain the fixed
   continuation. This freezes the selected experts and gate history while
   allowing the cache/placement policy to differ. Validate every event hash
   and tensor shape against the reference **before** including any timing.
   A different selected/weight/probability/demand hash invalidates the
   matched-route comparison; report it as ordinary fixed-continuation output.

The fixed-continuation arm alone cannot guarantee fixed routes: different
BF16 reduction order can change hidden states and router scores even with
identical token IDs. The frozen-router arm is therefore necessary if any
router hash diverges. Different cache capacities may still assign the same
global route to different ranks; measure that owner/packet change explicitly
rather than treating it as a route mismatch.

## Current decode hot path

`headline_ours_deepseek_worker.py` intercepts each routed MoE, computes
router probabilities, calls `LiveMetadata.collect`, runs `Policy.apply`,
builds and packs rank-partial indices, binds physical expert slots, enqueues
demand H2D, dispatches tokens, executes ready experts, and returns partials.
`LiveMetadata.collect` packs IDs and the probability tail, all-gathers it,
copies the gathered record to pinned CPU memory, then updates gate history.
The controller and layout builders are host-side. The controller receives
unit weights for admission demand; the **actual** local routing weights are
still scattered into the GPU dispatch payload before expert execution.
`PriorityH2DScheduler` transfers from the full pinned host store on a
separate stream. `DeepseekNativeExpertExecutor` makes one C++ call per ready
wave, but its C++ loop performs separate gate/up/down GEMMs per expert; it is
not grouped GEMM. `FusedTokenRankTransport` uses NCCL all-to-all for dispatch
and rank-partial return, followed by source-side `index_add_` combine.

These code facts nominate metadata D2H, host placement/layout, ready-wave
fragmentation, per-expert GEMM launches, and return arrival skew as possible
critical-path costs. They do **not** quantify which is dominant; D1's
matched-route comparison and D3's sampled profiler must do that. Qwen's
separate overlap ablation is supporting cross-model context, not a DeepSeek
causal measurement.
