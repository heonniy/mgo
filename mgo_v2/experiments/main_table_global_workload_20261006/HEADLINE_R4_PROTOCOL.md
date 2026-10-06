# R4 Headline Main-Table Protocol

Date: 2026-10-07
Branch: `codex/main-table-global-workload-20261006`

## Scope

Run the first complete four-system R4 headline panel before expanding the full
C30/C60 x B16/B64 x L256/L512 matrix.

Cells:
1. C30, Ours-local B16, global B64, input256, output64.
2. C30, Ours-local B64, global B256, input512, output64.

Physical GPUs: 0,1,4,5 only.

Systems:
- llama.cpp-layer
- DeepSpeed ZeRO-Inference
- MoE-Infinity
- Ours

All four systems must consume exactly the same global request manifest in each
cell. B/rank is only the Ours/DP presentation label; frameworks without rank-local
request semantics receive the full global batch.

## DeepSpeed naming correction

The MoE-Infinity paper (arXiv:2401.14361v2) describes its DeepSpeed baseline as
DeepSpeed-Inference configured for FastGen. The paper further characterizes that
baseline as dependency-based prefetching plus LRU caching for MoE expert buffers.

That is NOT identical to generic ZeRO-3 CPU parameter offload.

For Qwen3-30B-A3B, this benchmark therefore labels the compatibility baseline
explicitly as **DeepSpeed ZeRO-Inference (CPU parameter offload)**. Do not call it
the exact MoE-Infinity-paper FastGen baseline.

Rationale:
- current DeepSpeed ZeRO-Inference provides model-agnostic Stage-3 CPU parameter
  offload and pinned host parameter memory;
- current FastGen/MII public model-family support does not establish a stock
  Qwen3-MoE serving/offloading path equivalent to the paper's baseline;
- forcing a custom Qwen3 FastGen port would cease to be a stock baseline.

If a paper-faithful FastGen comparison is later required, use a model with stock
FastGen MoE support (e.g. Mixtral) as a separate cross-model experiment.

## Common workload

Model: Qwen3-30B-A3B-Instruct-2507
Precision: BF16
Decode: greedy, ignore EOS, exactly 64 new tokens
KV cache: GPU resident
CPU attention/KV offload: disabled
Dataset: frozen ShareGPT-long manifests
Warmup: one disjoint global batch, then retain each framework's normal cache state
Primary repeats: three unprofiled repeats; median reported

Common outer timing:
- TTFT: synchronized global release -> global-batch first-token readiness
- TPOT: decode critical-path wall time / token intervals
- E2E: synchronized release -> completion of token64
- Throughput: total generated output tokens / measured generation wall time

Resource columns:
- max peak HBM/GPU
- host RSS
- pinned host memory when observable

## llama.cpp

Use BF16 GGUF.
Use `--split-mode layer` only.
R4 visible GPUs: 0,1,4,5.
C30 mapping: `--n-cpu-moe 34`; 14/48 MoE layers remain GPU resident,
approximately 15.75 GiB expert weights, conservatively below Ours C30
16.198 GiB global expert budget.
The server receives global concurrency 64 or 256 according to the cell.
Disable prompt-prefix reuse for this benchmark.

## DeepSpeed ZeRO-Inference

Use Stage-3 CPU parameter offload:
- BF16
- offload_param.device=cpu
- offload_param.pin_memory=true
- no KV offload
- no weight quantization
- no custom Qwen3-specific kernel injection

Launch four ranks and partition the exact global manifest evenly:
- global64 -> 16 requests/rank
- global256 -> 64 requests/rank

Calibrate stage3 live/prefetch residency so measured peak GPU parameter
residency respects the C30-equivalent target. Record the calibrated knobs and
actual peak HBM. This is a generic parameter-streaming baseline, not an
expert-selective cache.

## MoE-Infinity

Use the stock Qwen3-MoE multi-GPU offloading path if the checked-out upstream
revision passes a correctness smoke test.
Expose GPUs 0,1,4,5 to one generic multi-GPU process and submit the full global
batch (64 or 256 requests).
Keep stock/default request-level tracing, expert prefetching, caching and pinned
I/O behavior enabled.
Calibrate actual aggregate GPU expert-cache bytes to <=16.198 GiB for C30.
Do not map C30 blindly to a generic device-memory ratio.

## Ours

Use the frozen selected runtime:
- Normal/H0 executor
- rank-private full-pinned expert source
- no H1/H1b graph cache
- C30 = 1,843 global expert slots
- V3 P2/T2 optimized overlap stack
- selected final placement policy from the policy-ablation branch when frozen
- prefill-created expert residency continues into decode

CPU pinned memory cost:
- 54 GiB/rank
- 216 GiB total for R4

This does not increase C30 GPU expert residency. Report pinned host memory
explicitly as a resource cost.

## Pinned-memory interpretation

The rank-private 216 GiB pinned backing is acceptable on the current ~2-TiB host
for this HBM-constrained study, and the implementation has already completed R4
physical validation. It is not free:
- page-locked pages are non-reclaimable/non-swappable;
- it raises the host-memory and memlock requirement;
- it scales to 432 GiB for R8.

Therefore the paper must not claim equal host-memory footprint across systems.
The fairness constraint is equal GPU expert-residency/HBM budget and identical
workload; each baseline uses its native host-offload implementation. Always
report host RSS and pinned-host bytes beside peak HBM.
