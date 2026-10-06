# Stage B5 — H1b post-expert communication barrier isolation

Status: implementation ready; no GPU run is authorized by this commit.

## Goal

Create a clean diagnostic execution order for H1b:

```
routing / admission
 -> async demand H2D
 -> forward A2A completion
 -> ready-first H2D + expert execution
 -> local current-stream completion
 -> GLOBAL rank barrier
 -> return A2A
 -> combine
```

The new barrier is **after all current-layer local expert work** and **before return communication**.

This mode is intended to separate:
- local rank-ready / execution imbalance;
- synchronization wait caused by the slowest rank;
- return communication after ranks are aligned.

It is not claimed to be the fastest production runtime.

## Runtime switch

Case field:

```json
"post_expert_barrier": true
```

Default is `false`, so existing V3/H0/H1b/H2 behavior is unchanged.

Barrier mode is applied only to fused decode MoE events (`index >= 48`).

## Exact synchronization semantics

After `moe.expert_compute`:

1. `moe.post_expert_local_complete`
   - synchronize the current compute stream;
   - this guarantees all current-layer expert gather/kernel/weight work and all required H2D wait-event dependencies have completed locally.

2. `moe.post_expert_global_barrier`
   - execute `dist.barrier()`;
   - synchronize the current stream after the barrier;
   - only then enter `moe.return_a2a`.

Future speculative prefetch on the dedicated H2D stream is **not drained**. The barrier isolates current-layer completion without turning the runtime into an all-prefetch-complete barrier.

For decode64 / 48 MoE layers, every rank must report exactly:

```
64 * 48 = 3072
```

post-expert barriers.

## Primary intended experiment

R4 GPUs 0,1,4,5 only.

C30 / local B128 / H1b / V3 P2-T2 / BF16 / substitution OFF.

Policies:
- BR
- FCA

First run correctness and physical-copy parity. Do not report timing if asynchronous prefetch cancellation changes H2D copy counts/bytes.

Then collect a separate Nsight diagnostic with the new NVTX ranges.

## Interpretation

The barrier moves wait caused by late rank completion out of return-NCCL residency and into an explicit synchronization point.

Do not interpret barrier duration as pure execute imbalance without subtracting/characterizing the barrier's own collective service cost.

Likewise, return A2A after the barrier is a cleaner communication measurement, but it still contains the actual split/payload-dependent collective service.

Primary quantities to report later:

```
T_local_complete
T_global_barrier
T_return_A2A
TPOT
```

and the TPOT-normalized exposed penalties derived from them.

## Safety / scope

- Keep H1b arithmetic, cache, admission, eviction, prefetch and transport semantics unchanged.
- No H2/grouped-GEMM requirement.
- No Stage C oracle.
- No C60/R8 automatic extension.
- Do not touch GPUs 2,3,6,7.
