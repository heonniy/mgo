# Eight-rank PCIe quota lookup for main_OURS

Use Qwen ShareGPT R8/C30/B16 per GPU/input512/output64, native C++
Ready-First expert execution, optimized prefill/decode layout, Near
placement and prefetch OFF. Keep the frozen workload and model unchanged.

Calibrate every one of the 255 nonempty physical-GPU subsets with a pinned
9 MiB Qwen-expert source for one-copy bursts. For 2, 4, 8 and 32-copy
bursts, only the all-eight subset is required by balanced R8 quotas.
Measure each case twice, then repeat the all-eight anchors and any
one-copy subset with >0.2 ms launch skew 16 times. Preserve all raw samples
outside Git; exclude only high-skew refinement samples from calibration.
The calibration has no model, NCCL or expert compute and therefore predicts
only H2D service. Recheck host/GPU memory and leave unrelated processes
untouched. Only the owned model loads on GPUs 0/1/4/5 may be stopped and
restored; GPUs 2/3/6/7 are already idle.

For each miss count 0..128, enumerate the balanced quota vectors (rank
max-min <=1). Use calibrated subset/rank transfer durations to minimize
predicted latest-rank H2D completion; break ties by sum of rank durations
then physical rank. The simple baseline assigns extra misses to the ranks
with the fastest measured all-eight 8-copy service. Precompute both 129x8
tables offline.
The compiled Numba policy reads one table row per layer; the existing Near
expert-placement rule, cache budget, eviction, transport and executor stay
unchanged. A small CPU test must verify every quota row and placement count.

Run one bounded functional smoke for each new policy, then two unfiltered
full-target repeats for original Near, fast-rank baseline and PCIe lookup.
If first-pair TPOT or E2E differs by >2%, add exactly one third repeat.
Report TTFT, TPOT, E2E, TPS, predicted H2D change, actual fetch counters,
output-token parity and controller overhead evidence. Do not claim the
microbenchmark model guarantees an end-to-end gain.
