# R4 new_OURS cache sweep

Measure `new_OURS` at C20/C30/C40/C50 on the existing frozen Qwen3-30B
ShareGPT R4 workload: physical GPUs 0/1/4/5, local batch 16, 512 input
tokens and 64 output tokens. Keep `LA_CA_NEAR`, pinned expert source,
prefetch OFF, native prefill, compiled dense routing, and the strict
`hit_then_miss` grouped decode path fixed. Cache capacities are respectively
1228, 1843, 2457, and 3072 expert slots across four ranks.

Each cell runs one warmup followed by two clean measurements, resetting the
expert cache before each measured batch. Report unfiltered TTFT, TPOT, and
global throughput (4096 generated tokens / E2E seconds). Show both samples,
their mean and range; flag a cell if either TTFT or TPOT differs by more than
5% of its two-sample mean. Do not select favorable repeats. Use the guarded
R4 launcher with one job at a time, host/HBM/temperature checks, and restore
the eight owned model loads between jobs. Preserve raw outputs outside Git.

The Qwen C20–C50 MoE-Infinity, DeepSpeed, and synchronous balanced llama.cpp
R4 baseline measurements already exist in
`../qwen_cache_ablation_20261009/BASELINE_SWEEP_RESULTS.json`.
Compare their validated medians against the new measurements, while marking
that the baselines were recorded on earlier dates and use different runtime
implementations. `new_OURS` is a grouped decode candidate, not the selected
`main_OURS` implementation in the prior baseline comparison.
