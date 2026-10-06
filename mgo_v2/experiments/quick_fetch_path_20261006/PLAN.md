# Quick R4 Expert-Fetch Path Check

Purpose: answer two implementation questions before the main table without a full model run.

1. When GPUs 0,1,4,5 fetch experts simultaneously, does a specific rank suffer disproportionate slowdown?
2. How much latency is removable by eliminating the current pageable/file-backed -> pinned staging copy?

## Compared paths
- current_staging: existing PriorityH2DScheduler, 2 pinned stages/rank.
- direct_pinned: 32 selected experts copied once into a 288 MiB pinned pool/rank before measurement, then copied directly to GPU.

The direct-pinned path is an upper-bound microbenchmark. It does NOT pin the full 54 GiB expert store.

## Conditions
- isolated logical ranks 0/1/2/3 (physical 0/1/4/5)
- pair01 and pair23
- all4
- burst sizes 1,8,16,32 experts
- 1 warmup + 3 measured repeats

No model forward, no GEMM, no NCCL payload communication.

Primary metrics:
- burst wall time per active rank
- max/critical-rank burst wall time per repeat
- current staging queue/stage/DMA decomposition
- direct-pinned gain over current staging

Interpretation:
- all4 slowdown that is similar on every rank => shared PCIe/host-memory saturation.
- persistent one-rank slowdown => rank/NUMA/path asymmetry worth accounting for.
- >=10% direct-pinned critical-path gain => consider a registered/pinned expert-store path in Ours.
- <5% gain => keep bounded staging; full-store pinning is unlikely to matter.

Safety: physical GPUs 0,1,4,5 only. Never touch 2,3,6,7.
Runtime bound: 10 minutes; expected to finish much sooner because no model is loaded.
