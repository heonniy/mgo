# Qwen ShareGPT R8 main-system comparison

Use one eight-rank experiment on physical GPUs 0–7. Each rank handles B16,
so the global batch is 128 requests. Use the first 128 ShareGPT requests from
the prior frozen Qwen source pool and a disjoint 128-request warmup; input is
the same exact 512 token IDs and generation is 64 BF16 greedy tokens, ignoring
EOS. Commit hashes and timing summaries only, never raw token manifests.

Hold the C30 global expert capacity at 1,843 nine-MiB expert slots and split
it as 231/231/231/230/230/230/230/230. Run selected main_OURS (Near,
native expert execution, compiled layouts, full pinned source, prefetch OFF),
repaired MoE-Infinity, audited DeepSpeed ZeRO-Inference, and synchronous
llama.cpp with CUDA graphs/reuse OFF. The llama.cpp baseline uses balanced
one-whole-expert-layer-per-GPU placement; its layer granularity leaves part
of the nominal C30 capacity unused, which must be reported explicitly.

Run one system at a time. First pass a small functional smoke for each
system, then perform two clean full target repeats per system after a disjoint
warmup and cache reset. If either TPOT or E2E differs by >2% but ≤5% between
the two targets, allow exactly one third target; if >5%, mark unstable and
do not add open-ended repeats. Preserve all raw attempts and report median
and full range for TTFT, TPOT, E2E, and throughput. Do not substitute R4
results or treat two parallel R4 jobs as R8.

The guarded launcher stops only the eight owned model-inference loads, aborts
if an external GPU process is present, requires at least 384 GiB available
host memory before launch and 96 GiB throughout, and stops below 2 GiB free
GPU memory or at 85 C. Restore the model-inference loads after each job,
including failures. The private root is
`/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009`.
