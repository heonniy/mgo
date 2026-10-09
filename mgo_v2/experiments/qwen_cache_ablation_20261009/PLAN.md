# Qwen3-30B main_OURS cache-capacity comparison

Test whether Qwen3-30B-A3B-Instruct-2507 shows the same weak TPOT response to
larger expert caches observed with DeepSeek-V2-Lite. Use the frozen ShareGPT
Qwen B16/input512/output64 requests from the completed two-model main table.
Run R4 only on physical GPUs 0/1/4/5. Compare C20/C30/C40/C50, one guarded
job at a time. Use **main_OURS only**: native C++ expert execution, compiled
prefill/decode layout, Near placement, full pinned host expert source, and
prefetch OFF. Clear expert cache after warmup, then take three unfiltered
target repeats per cache setting. Report median and full range for TTFT, TPOT,
and E2E; also record per-rank H2D bytes and request/token stability.

The model has 48 routed layers × 128 experts = 6,144 expert IDs, each 9 MiB.
Floor the global expert-slot budget: C20=1,228 (307 per rank), C30=1,843
(461/461/461/460), C40=2,457 (615/614/614/614), and C50=3,072 (768 per
rank). The selected runtime reserves two physical slots per rank outside its
logical MAIN capacity, so physical expert residency stays within the budget.
The rank-private full pinned host store is 54 GiB per rank; verify host
headroom and the supervisor's HBM stop guard before timing. No other GPU may
be used.

Keep all four cache settings on identical frozen input requests and rank
order. Use a separate, non-primary generation diagnostic at the endpoints
(C20/C50) if needed to attribute exposed H2D wait and expert/communication
spans; require token/cache parity with the timed run. Raw token IDs and phase
traces remain outside Git. Commit validated aggregate rows as jobs finish,
repair failed attempts under new labels, and restore owner model inference
loads after guarded jobs. Compare Qwen's curve with the completed DeepSeek
B16/L512/O64 result, but do not claim cross-model equality of routing demand.
