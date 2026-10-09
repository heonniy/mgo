# C20/C50 TPOT plateau attribution

The completed 16-cell ablation shows that main_OURS transfers 36.4% fewer
expert bytes at C50 than C20, while TPOT improves only 2.15%. Total H2D bytes
do not measure exposed latency. Run a separate, instrumented comparison at
C20 and C50 on the same frozen ShareGPT B16/input512/output64 requests to
measure which part of H2D remains on the rank's decode path. This is a
diagnostic follow-up, not a replacement for the three-repeat primary table.

Use the existing guarded supervisor on physical GPUs 0/1/4/5. For each cache
setting, execute one disjoint warmup and one target. Enable
`MGO_DEEPSEEK_CACHE_DIAG=1` only for these attempts. Per rank and decode step,
record H2D bytes, native expert groups/waves, the number of not-ready expert
waits, main CUDA-stream H2D wait-event spans, host time spent submitting those
waits, route-to-dispatch host span, dispatch span, expert span, return/combine
span, and full rank-step wall time. Use CUDA events without extra per-layer
synchronization; the existing end-of-step synchronization resolves events.

Compare the 63 decode steps separately from prefill. Report median and total
per-rank waits and spans, plus the maximum-rank values that can affect TPOT.
Do not sum overlapping spans into a fictitious end-to-end decomposition. If
diagnostic TPOT shifts materially from the primary, treat only its component
ratios and event relationships as evidence. Keep raw tokens and per-step traces
outside Git; commit aggregate attribution and its limitations.
