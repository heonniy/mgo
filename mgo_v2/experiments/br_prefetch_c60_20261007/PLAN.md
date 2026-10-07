# BR C60 prefetch physical ablation

Owner scope reduced to B8/B16/B64 x BR x prefetch OFF/ON only (six cells).
R4 GPUs0/1/4/5; input256; cold cache/history before prefill, then256 decode
forwards (257 generated tokens). Gate-W128 eviction, seed42, original decode
layout, optimized prefill, H0/full-pinned. Same target prompts as prior study.
One disjoint8-decode warmup, one clean full primary and one separate full
phase diagnostic per arm. No automatic repetition or other admission policies.
Model and rank-private pinned sources reused across two arms in each batch.
Both use3678 MAIN +8 reserved prefetch slots. OFF disables prefetch_next
and leaves those8 slots unused; ON uses them. Total physical capacity
922/922/921/921 and MAIN capacity920/920/919/919 are identical. Both retain streaming demand-H2D overlap,
ready-first and T2 synchronization; OFF is NOT the old P0 barrier arm or full-MAIN capacity control.
Batch16 runs ON then OFF; others OFF then ON. Single-shot limits remain.

Primary TPOT=(max-rank final completion - max-rank first completion)/256.
Primary has only existing checks/step timing and lightweight prefill byte
snapshots, no detailed diagnostic timers. Payload volume is remote dispatch
plus return send bytes, excluding local traffic, routing metadata and NCCL
protocol overhead. Report H2D bytes separately.

Diagnostic MoE time uses whole MLP CUDA-event boundaries, including router,
metadata, placement, dispatch, H2D dependencies, expert execution, partial,
return and combine; excludes attention/layernorm outside MLP. Main aggregate:
mean over256 steps of max-rank sum of48 MoE layer durations. Also report all
rank totals. This diagnostic proxy is not subtracted from clean primary TPOT.
Detailed spans include CPU submission gaps and peer waiting. H2D copy-stream
service is non-additive. Save host monotonic timestamps for collective-entry
skew, per-layer expert counts/token-expert rows and required H2D waits.
Diagnostic tokens/cache/bytes must match each arm's primary. OFF/ON token
agreement is reported; native routing may differ under BF16 accumulation.
Host guard384/96GiB,85% per-rank HBM cap. No other GPUs or new matrix.
