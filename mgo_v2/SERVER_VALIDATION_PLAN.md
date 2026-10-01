# Current-server bring-up plan

This plan is for implementation validation after the policy code lands.

## Execution status — 2026-10-01

The server run and its limits are recorded in
[SERVER_VALIDATION_RESULTS.md](SERVER_VALIDATION_RESULTS.md), with compact
measurements under [results/server_20261001](results/server_20261001).

| Stage | Executed scope |
|---|---|
| 0 | Eight H100s; one visible GPU per process; strict binding to the sole OS-visible NUMA node; legacy controllers disabled. |
| 1 | Native R1/R4/R8 output/router/token parity at 10% cache, plus padded R4 batch-8 parity; all exact checks pass. The small cache also exercises replacement. |
| 2 | Per-layer slot/address/single-copy/fetch audits and independent FFN fixtures, including empty ranks, custom streams and pool reset, pass. |
| 3 | .20/.65 substitution counters, source/target maps and gate mass match recorded policy replay. |
| 4 | LRU/Gate/Coverage replay passes; both R4/R8 ablations include actual fetches, bytes, reloads and paired short-screen quality. |
| 5 | All seven admission policies measured at R4/R8, including submitted NCCL tensor bytes, CUDA collective intervals, controller time and rank-load CV. Full-model Nsight audits also verify actual expert transfers and NCCL kernel intervals in all 120 rank/cell ranges; byte counters exclude wire overhead. |
| 6 | Direct slot views eliminate expert D2D copies in the Nsight fixture; H2D/compute overlap and R1/R2/R4/R8 H2D/staging/NVSwitch bandwidth measured. Physical remote-NUMA comparison is unavailable in this VM. |
| 7 | All 60 conditions × 2 repeats complete, with 720 audited rank receipts and batch-matched native controls. Tables cover the requested batch/cache grid and A/B/C ablations. |

Stage 7 is a bounded **16-step server measurement**, not a publication-ready
accuracy or significance claim. Two repeats expose timing variation but do
not establish small speedups. The numeric quality screen uses held-out
GSM8K-train questions, a new zero-shot prompt and a 16-token cap; standard
few-shot/test accuracy and longer-run serving behavior remain unmeasured.
The original stage requirements below remain the research checklist.

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
