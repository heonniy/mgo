# DeepSeek-V2-Lite ShareGPT cache-capacity ablation

**Complete: 16/16 system cells, 48/48 target repeats passed.** R4 used only
physical GPUs 0, 1, 4, and 5. Each rank served B16; each request had 512 input
tokens and 64 generated tokens. Every cache setting and system used the same
frozen 64-request ShareGPT target batch, a separate 64-request warmup batch,
and three unfiltered target repeats. The table shows the median of the three
targets. [PROGRESS.md](PROGRESS.md) and [RESULTS.csv](RESULTS.csv) retain every
metric's minimum and maximum.

| Expert cache | System | TTFT (s) | TPOT (s/token) | E2E (s) |
|---:|---|---:|---:|---:|
| 20% | main_OURS | 0.572 | 0.2879 | 18.737 |
| 20% | MoE-Infinity (repaired) | 1.619 | 1.2598 | 81.096 |
| 20% | DeepSpeed ZeRO-Inference | 4.853 | 4.7116 | 301.984 |
| 20% | llama.cpp balanced | 145.206 | 0.4134 | 171.265 |
| 30% | main_OURS | 0.603 | 0.2847 | 18.617 |
| 30% | MoE-Infinity (repaired) | 1.674 | 1.2591 | 81.010 |
| 30% | DeepSpeed ZeRO-Inference | 4.907 | 4.6384 | 297.022 |
| 30% | llama.cpp balanced | 145.381 | 0.4135 | 171.452 |
| 40% | main_OURS | 0.588 | 0.2817 | 18.604 |
| 40% | MoE-Infinity (repaired) | 1.644 | 1.2169 | 78.311 |
| 40% | DeepSpeed ZeRO-Inference | 4.795 | 4.6506 | 297.785 |
| 40% | llama.cpp balanced | 118.479 | 0.3485 | 140.430 |
| 50% | main_OURS | 0.612 | 0.2817 | 18.360 |
| 50% | MoE-Infinity (repaired) | 1.624 | 1.2399 | 79.824 |
| 50% | DeepSpeed ZeRO-Inference | 4.827 | 4.6891 | 301.390 |
| 50% | llama.cpp balanced | 98.434 | 0.2816 | 116.231 |

main_OURS was the fastest E2E system at every cache size. At C50, its TPOT and
llama.cpp's TPOT were effectively tied: their medians were 0.2817 and 0.2816
s/token, respectively, with overlapping repeat ranges. Their TTFT values were
0.612 s and 98.434 s, respectively.
Its C50 E2E was 18.360 s versus 79.824 s for MoE-Infinity, 116.231 s for
llama.cpp, and 301.390 s for DeepSpeed. main_OURS used the DeepSeek-specific
Near placement, native expert executor, and prefetch OFF.

From C20 to C50, main_OURS' measured-batch H2D traffic fell 36.4% (1,424.3
to 905.3 GiB summed across four ranks), while median TPOT improved only 2.15%
(0.2879 to 0.2817 s/token). C40 and C50 TPOT medians were nearly identical.
This is an observed runtime trend, not proof that H2D is off the critical
path: generated continuations can change with capacity. MoE-Infinity's EAM
evictions fell from 91,400 to 59,863, but its fastest median TPOT occurred at
C40 rather than C50. [CACHE_ACTIVITY.md](CACHE_ACTIVITY.md) has the separate
traffic and eviction counters; they are not cross-system-equivalent misses.

The llama.cpp expert placement was deliberately quantized to whole layers:
1/1/1/1 GPU expert layers at C20 and C30, 2/2/2/2 at C40, and 3/3/3/3 at C50.
Its C20/C30 resident placement was therefore identical despite different
budgets. C50's larger placement reduced its TPOT substantially, but its long
prefill still dominated E2E. DeepSpeed's expert budget is a parameter-residency
constraint; it is not a bound on total HBM including dense weights and runtime
buffers.

The input request batch was frozen, but the future decode route was not.
main_OURS produced identical full 64-token outputs across its three repeats
within each cache setting; C20 and C50 matched for only 25/64 complete output
sequences. MoE-Infinity and DeepSpeed also had repeat-level token differences;
llama.cpp did not within a cache setting.
[TOKEN_STABILITY.md](TOKEN_STABILITY.md) records the agreement counts. Hence
these are real greedy-run performance results at each capacity, not a
bit-identical frozen-continuation causal measurement.

All 16 guarded jobs passed their capacity and output checks. The minimum
sampled host-available memory was 1,642.8 GiB and the largest sampled owner
GPU HBM use during a job was 13,693 MiB; no OOM or failed full attempt
occurred. GPUs 2, 3, 6, and 7 were untouched. Owner model inference loads
were restored on GPUs 0, 1, 4, and 5 after the queue. Raw requests and token
IDs remain outside Git; aggregate results, hashes, and checks are in this
experiment folder.
