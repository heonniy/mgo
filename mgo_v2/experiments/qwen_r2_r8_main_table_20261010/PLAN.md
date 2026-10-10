# Qwen R2/R8 quiet-host main-table extension

Use Qwen3-30B, ShareGPT, input512, 64 greedy output tokens, per-rank B16,
C30 expert cache, BF16. Measure `main_OURS` (Near, native Ready-First C++
expert executor, compiled prefill/decode indices, direct pinned CPU source,
prefetch OFF), MoE-Infinity, DeepSpeed ZeRO-Inference, and synchronous
balanced llama.cpp. Report TTFT, full TPOT, E2E and global TPS. Do not use
grouped `new_OURS` as the OURS row.

R8 uses the already frozen 128-request manifest in
`/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/WORKLOADS.json`,
all GPUs 0–7, and the established audited workers. The user stopped the
four owned model-inference loads on GPUs 2/3/6/7; keep them stopped between
jobs. The guarded launcher pauses/restores only the four remaining owned
loads on GPUs 0/1/4/5 and rejects any foreign GPU process. Preserve prior
R8 job attempts; use new labels.

R2 uses GPUs 0/1, the first 32 target and disjoint 32 warmup requests from
the same frozen source, with global C30 capacity of 1,843 expert slots split
922/921. Pause the owned inference loads on GPUs 4/5 during each R2 job as
well, then restore them; GPUs 2/3/6/7 remain idle. Do not compare R2 and R8 as equal global-batch throughput: per-rank
B16 makes their global batches 32 and 128. Verify the four workers support
two GPUs, budget checks, local batch semantics and synchronous llama.cpp
balanced expert-layer placement before timing.

Run a bounded functional smoke first for each worker and rank count. Then
use two unfiltered clean target repeats per cell after disjoint warmup and
cache reset. The 2026-10-10 follow-up requests exactly one third repeat for
any cell whose first-pair TPOT or E2E differs by >2%, including pairs above
5%; stable cells stop after two. Report the median of three, preserve the
full range, and flag any >5% initial-pair gap rather than treating the
median as a stable point estimate. Keep raw receipts outside Git and commit
only a provenance-backed summary.

The same follow-up requested one additional R2 main_OURS check after its
original three-run primary result. Preserve that fourth run as an explicit
confirmation receipt; do not silently change the original three-run median.

The R8 MoE-Infinity long target stalled after four tokens while the warmup
completed. An eight-token diagnostic target queued 591,448 EAM speculative
candidate transfers across 384 layer calls; its warmup queued zero. For the
R2/R8 comparison, retain EAM cache budget and eviction priorities but
disable speculative transfer admission in both rank counts, as already
required for the DeepSeek baseline. A token progress receipt and external
120-second no-progress guard are included so a native wait cannot silently
consume the queue. Preserve the failing raw attempts and exclude diagnostic
timings from the main table.
