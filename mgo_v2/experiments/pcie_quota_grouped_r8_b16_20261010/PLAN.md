# R8 PCIe quota with grouped new_OURS

Use the established Qwen3-30B ShareGPT R8/C30/local-B16/input512/output64
workload on physical GPUs 0–7. Compare original Near, fastest-rank balanced
quota, and the calibrated PCIe balanced-quota lookup with exactly the same
`new_OURS` decode: native prefill and expert backend, compiled dense/layout
paths, prefetch OFF, grouped resident hits first, wait for all demand misses,
then group all misses. Do not substitute the older `two_wave` C arm.

Use the previously frozen Near next-token IDs for the initial comparison.
This controls decode input-token drift while preserving actual argmax and
rank fetch receipts; BF16 hidden states and router choices can still differ.
One guarded smoke per policy precedes two unfiltered target repeats. Add
exactly one third repeat if the first two differ by more than 2% in TPOT or
E2E. Report all samples, full range, quota copies by rank, token differences,
and source commit. Preserve raw jobs outside Git. The guard pauses and
restores owned loads on GPUs 0/1/4/5, leaves 2/3/6/7 idle, and checks host,
GPU memory, temperature, and foreign processes.

If the PCIe lookup does not show a clear TPOT gain over the simple fast-rank
quota, run a bounded ShareGPT sample-seed search with all eight GPUs under
the same grouped mode. Freeze the candidate list before measuring, screen
all candidates without dropping unfavorable repeats, and confirm the best
observed seed in a fresh job. A selected seed is a workload example, not
evidence of general advantage; report all screened seeds and any failed
confirmation.
