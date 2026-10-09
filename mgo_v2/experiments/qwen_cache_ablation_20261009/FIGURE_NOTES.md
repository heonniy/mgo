# Qwen cache sweep figures

`figures/qwen_cache_systems.png` and `.pdf` show four systems on the frozen
ShareGPT Qwen3-30B, R4 (physical GPUs 0/1/4/5), B16 per rank, input512,
output64 workload at C20/C30/C40/C50. For OURS, the source is
`TIMING_RECHECK.json`: the two unfiltered post-cleanup repeats endorsed by
`PROGRESS.md` for the current capacity comparison. Using those same repeats
for all three metrics keeps TTFT, TPOT and throughput coherent. The original
chronological three-repeat OURS data remain in `PROGRESS.md` and are not
silently blended into the figure. The other three systems come from validated
`BASELINE_SWEEP_RESULTS.json`; C30 is a validated earlier three-repeat run,
while the new capacities have two repeats unless the bounded stability rule
requires a third. Bars are medians and whiskers span all clean observations.

**TPOT** is seconds per generated output token. **TTFT** is seconds until the
first token. **TPS** is total generated output tokens divided by complete
batch E2E time: 64 requests × 64 output tokens / E2E seconds. Thus TPS
includes prefill time and is not the reciprocal of TPOT. The llama.cpp TTFT
panel uses a separate, explicit scale because it is far above the other
systems; no bar axis is truncated. The C20–C50 results are separate runs, not
simultaneously paired, and the OURS TTFT two-repeat ranges are particularly
wide. Source medians, full ranges, and sample counts are in
`figures/performance_source.csv` and `.json`.

`figures/two_model_eviction_hit_rate.png` and `.pdf` show MAIN distinct-expert
decode hit rate at C30/C60 for gate-score, LFU with retained history, and LRU
with retained history. Qwen uses the prior fixed B16/input256/256-decode
no-prefetch CPU replay in `main_eviction_history_20261007`. DeepSeek uses
the new CPU-only replay of the previously captured D1 B16/input512/63-decode
route files in `deepseek_eviction_hit_20261009`. The workloads and trace
lengths differ by model; compare policies within each model rather than
absolute hit rates between models. C60 is a CPU counterfactual in both cases,
not a physical timing result. The figure never implies that hit rate equals
end-to-end performance improvement. Exact values are in
`figures/eviction_hit_rate_source.csv`.

Render with `python mgo_v2/scripts/plot_cache_sweep_figures.py` after all
twelve Qwen baseline rows pass validation.
