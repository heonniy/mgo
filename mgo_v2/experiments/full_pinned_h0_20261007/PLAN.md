# Final runtime: Normal/H0 with full-pinned expert sources

Owner decision (2026-10-07): final runtime uses Normal/H0, not H1b graph execution.
Keep exact C30/C60 expert cache budgets; remove persistent graph input/output buffers.
Full pinned sources remain rank-private CPU memory: 54 GiB/rank, 216 GiB for R4.

Validate R4 GPUs 0,1,4,5 only, C30, local B128, frozen existing requests,
BR/P2/T2, BF16, decode64, V3 overlap, unique combine and async metadata.
Run one correctness pass and two counterbalanced timing repeats of H0 STAGED
and H0 FULL_PINNED. STAGED is a diagnostic control, not a final runtime option.
No H1b graph construction or fallback. Compare exact tokens/controller state/
copy bytes/transport between source modes. Keep all timing samples and flag
>5% spread. No automatic repeat expansion or new workload search.

Expose the selected runtime through the existing factory, with H0 and
full-pinned source configured explicitly and allocated outside timing.
Do not label the default physically validated until this run passes.
Report process high-water HBM/RSS, 216 GiB CPU pin cost and C30 residency
separately. C60 is a supported cache budget, not physically validated here.
Follow the 384/96 GiB launch/abort host guards and 30-minute run bound.
Restore only owned model loads on 0,1,4,5 after completion.
