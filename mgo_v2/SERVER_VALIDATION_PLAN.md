# Current-server bring-up plan

This plan is for implementation validation after the policy code lands.

## Stage 0 — environment

- one process per GPU;
- call `pin_rank_before_cuda_import()` before importing torch/moe_infinity;
- R1 first, then R4, then R8;
- inspect `nvidia-smi topo -m` and detected GPU->NUMA mapping;
- use strict local NUMA binding initially;
- disable legacy autonomous eviction/prefetch.

## Stage 1 — native exact parity

No substitution. Unlimited or very large expert cache.

Compare against native Qwen3 on fixed prompts:
- selected experts;
- routing weights;
- per-layer output checksum;
- generated token ids.

Do not proceed on unexplained divergence.

## Stage 2 — physical slot parity

Enable fixed slot capacity but use:
- random admission;
- LRU eviction;
- no substitution.

After every layer in debug mode:
- Python owner/slot map == C++ get_cached_experts;
- no duplicate expert copies;
- no active-slot overwrite;
- H2D fetch count == logical miss count.

## Stage 3 — A substitution parity

Enable .20/.65 expert-level substitution.

Compare model-execution counters to the validated logical experiment:
- Expert Hit;
- Expert SubHit;
- Expert Miss;
- source->target mappings;
- substituted gate mass.

## Stage 4 — B eviction table

At fixed admission baseline, run:
- LRU;
- Gate W128;
- Coverage W128/k1/lambda2.

Collect:
- expert-level Hit/SubHit/Miss;
- distinct H2D fetches;
- host bytes;
- reloads;
- accuracy.

## Stage 5 — C admission table

Freeze B and compare:
- balanced random;
- greedy current;
- greedy current+path;
- Hungarian current;
- Hungarian + same-layer;
- Hungarian + same-layer + path;
- swap-refined winner.

Collect:
- deduplicated remote token->rank pairs;
- actual NCCL bytes/time;
- controller time;
- rank token CV;
- H2D fetches;
- accuracy.

## Stage 6 — low-level performance cleanup

Before final speedup claims:
- remove/avoid resident-slot -> MoEMLP full-expert D2D copy on every hit;
- profile H2D direct vs staging;
- verify overlap with Nsight Systems;
- measure PCIe contention at R2/R4/R8;
- verify local-NUMA H2D bandwidth per rank;
- check NVSwitch all-to-all bandwidth.

## Stage 7 — paper tables

Only after Stages 1-6 pass:
- TTFT / TPOT / throughput;
- R4/R8;
- local batch 4/8/16/32;
- cache ratio 10/20/30/40/50%;
- A/B/C ablations;
- accuracy;
- controller overhead.
