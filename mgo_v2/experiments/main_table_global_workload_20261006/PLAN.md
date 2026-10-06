# Main-Table Multi-GPU Offloading Benchmark Plan

Status: DRAFT BENCHMARK PLAN
Date: 2026-10-06
Branch: `codex/main-table-global-workload-20261006`

## Goal
Produce the draft paper main table for multi-GPU MoE inference under constrained GPU expert-memory budgets.

Systems:
1. llama.cpp, multi-GPU `split-mode=layer` only.
2. DeepSpeed ZeRO-Inference / ZeRO-3 CPU parameter offload.
3. MoE-Infinity stock Qwen3-MoE multi-GPU offloading.
4. Ours, current optimized multi-rank offloading runtime.

This main-table study is separate from the strict-phase BR/LA/LA+CA placement characterization. The strict H2D/compute barriers are NOT used for the production main table. Ours uses the optimized runtime, and the cache/placement state created during prefill naturally continues into decode.

## Common model and numerics
- Model: Qwen3-30B-A3B-Instruct-2507.
- Precision: BF16-equivalent weights/activations wherever supported.
- llama.cpp primary table uses BF16 GGUF, not Q4/Q8.
- Greedy decoding.
- Ignore EOS for timing and generate exactly 64 output tokens.
- Model load, tokenizer init, compile/JIT and one-time allocation are outside timing.
- CPU expert/parameter pages are prefaulted/resident before timing.
- KV stays on GPU; CPU attention/KV offload are disabled.
- Record exact upstream git SHA/package version for every baseline.

## Dataset and request construction
Primary dataset: existing ShareGPT V3 cleaned long-conversation pool.

Source pool:
- 2,048 frozen conversations.
- Qwen chat template already applied for the project workload.
- Source conversations have at least 512 valid prompt tokens.

Prompt lengths:
- L256: most recent 256 valid prompt tokens.
- L512: most recent 512 valid prompt tokens.

No short-prompt padding. Every request has exactly the target valid-token count.

Timed unit:
`prefill(input L) -> generate exactly 64 tokens`.

All systems receive the same request IDs, token IDs, ordering, and generation length for a cell.

## Global-workload semantics
B16/B64 remain Ours-local-batch labels, but the cross-system fairness unit is the corresponding GLOBAL request count.

| GPUs | Ours local B | Global requests |
|---:|---:|---:|
| 4 | 16 | 64 |
| 4 | 64 | 256 |
| 8 | 16 | 128 |
| 8 | 64 | 512 |

Create one immutable global manifest per (R, B, L) cell.

Framework consumption:
- Ours: evenly partition the global manifest by origin rank; existing EP+DP runtime. Prefill placement/cache state continues into decode.
- DeepSpeed: evenly partition the same global manifest across launched ranks; the union must exactly equal the global manifest.
- MoE-Infinity: generic Qwen3 multi-GPU path receives the full global request batch through its single multi-GPU process.
- llama.cpp layer split: one multi-GPU server receives the full global concurrent request set.

No baseline receives fewer global requests than Ours.

## Expert-GPU memory budgets
C30/C60 are GPU-resident expert-weight budget equivalents, not an identical cache knob in every framework.

Qwen3-30B-A3B expert store used here:
- 48 layers x 128 experts = 6,144 experts.
- Expert size = 9,437,184 bytes = 9 MiB.
- Full sparse expert store = 54 GiB.

Exact global budgets:
- C30: 1,843 slots = 17,392,730,112 bytes = 16.198 GiB.
- C60: 3,686 slots = 34,785,460,224 bytes = 32.396 GiB.

Ours slot distribution:
- R4 C30: [461,461,461,460]
- R4 C60: [922,922,921,921]
- R8 C30: [231,231,231,230,230,230,230,230]
- R8 C60: [461,461,461,461,461,461,460,460]

Dense weights, KV, activations and framework workspace are outside this percentage and are reported separately as total peak HBM.

## Baseline mapping

### llama.cpp
- Fix `--split-mode layer`; do not tune row/tensor in the draft.
- BF16 GGUF.
- C30: `--n-cpu-moe 34` -> 14/48 MoE layers on GPU -> 15.75 GiB expert weight.
- C60: `--n-cpu-moe 20` -> 28/48 MoE layers on GPU -> 31.50 GiB expert weight.
- Equal multi-GPU split unless correctness requires otherwise.
- No exhaustive tuning.

### MoE-Infinity
- Stock Qwen3-MoE multi-GPU offloading path.
- Keep normal stock/default cache + prefetch behavior.
- Calibrate actual GPU expert residency to <=16.198 GiB global (C30) or <=32.396 GiB global (C60).
- The byte budget is authoritative; do not blindly map C30/C60 to a generic device_memory_ratio.
- No workload-specific predictor/threshold tuning for the draft.

### DeepSpeed
- BF16 ZeRO-3 / ZeRO-Inference CPU parameter offload.
- Pinned CPU parameter memory where supported.
- No custom Qwen3-MoE expert cache or custom kernel injection unless a stock verified path exists.
- Calibrate live GPU parameter residency to the C30/C60 equivalent budget and record actual peak HBM.
- Only minimal tuning needed for budget compliance and correctness; no exhaustive sweep.

### Ours
Use optimized runtime, not strict characterization runtime.
- Exact C30/C60 expert slots.
- Prefill admission/placement enabled.
- Prefill-created cache state naturally continues into decode.
- Intended production overlap/prefetch/ready features may be enabled.
- No substitution/replication unless explicitly promoted into the final system definition.
- Record controller overhead separately.

## Physical matrix
- R4 / R8.
- C30 / C60.
- Ours local B16 / B64.
- Input L256 / L512.
- Output O64.
- Four systems.

Total: 2 x 2 x 2 x 2 x 4 = 64 system cells.

Draft priority:
1. R4/C30/B64/L512.
2. R4/C30/B64/L256.
3. R4/C30/B16/L512.
4. R4/C60/B64/L512.
5. Fill remaining R4 cells.
6. R8 only on an explicitly authorized 8-GPU allocation.

Current shared-server safety:
- R4 physical GPUs: 0,1,4,5 only.
- Never touch 2,3,6,7 on the current shared server.
- R8 must require an explicit authorized GPU list; never default to 0..7.

## Warmup and measured state
This is a system main-table benchmark, not cold-cache characterization.

For each system/config:
1. load model outside timing;
2. prefault CPU backing;
3. finish compile/JIT/alloc warmup;
4. run one untimed warmup global batch using disjoint request IDs;
5. keep the framework's normal residency/cache state;
6. measure the target global batch.

Measured target requests are not reused as warmup requests.

Prefill and decode form one continuous lifecycle; state from measured prefill is retained into measured decode.

## Metrics
Primary:
- TTFT: common outer wall-clock release to first-token readiness for the whole global batch critical path.
- TPOT: decode wall time after first-token readiness divided by the remaining 63 token intervals.
- E2E: release to completion of token 64 for the whole global batch.
- Throughput: generated output tokens / generation wall time.

Resources:
- peak HBM per GPU and max across GPUs;
- CPU resident memory;
- CPU->GPU transferred bytes when observable;
- expert-cache hit/miss for Ours/MoE-Infinity when observable;
- controller/framework CPU overhead;
- correctness and finite-output checks.

Common outer wall-clock timing is authoritative; do not compare incompatible framework-internal timers directly.

## Repeats
- one correctness/warmup run;
- three unprofiled measured repeats;
- median is the table value;
- record min/max/CV;
- >5% repeat spread => unstable, re-measure before headline use;
- profiler/diagnostic runs are separate.

## Draft vs final
Allowed for draft:
- Ours uses intended optimized runtime.
- llama.cpp fixed to standard layer split.
- DeepSpeed/MoE-Infinity use reasonable stock/default configs with only budget calibration.
- Headline first probe can be R4/C30/B64/L512.

Not allowed:
- disabling a baseline's normal cache/prefetch only to slow it down;
- giving a baseline fewer global requests;
- using less HBM for baselines than Ours;
- reporting known broken/misconfigured baselines.

Before paper-final numbers, revisit baseline versions and tuning.
