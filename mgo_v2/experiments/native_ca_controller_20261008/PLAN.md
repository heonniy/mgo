# Native CA controller candidate

The selected `main_OURS` remains native C++ expert execution, compiled
prefill/decode layouts, Near placement and prefetch OFF. Optimize only CA's
current-layer admission assignment. The original Numba/Hungarian CA policy
remains available as `CA`; the opt-in `CA_NATIVE` uses a C++ min-cost-flow
kernel over expert-to-rank edges and exact per-rank quotas. Both maximize the
same integer local distinct-demand objective. Equal-optimum assignments may
differ, so never claim cache or byte parity across these policies without
physical evidence. The native library is compiled under a lock before timed
generation; the legacy JIT function and its cache stay unchanged.

First compare native and legacy objective, quotas and determinism on bounded
CPU matrices for R1/R2/R4/R8 and 0–128 misses. Then use R4/C30/B16/input256/
output64 on GPUs0/1/4/5, full-pinned CPU experts, same native expert executor,
BF16 rank partials and prefetch OFF. Warm both policies, capture one
unrestricted Near route and teacher stream, and cold-reset each measured
policy. Run CA/CA_NATIVE/CA_NATIVE/CA on the frozen 63-decode trace with two
unfiltered primary repetitions each. Do a separate 16-decode diagnostic per
policy to measure placement controller CPU span. Require finite logits,
same-policy cache/controller/byte parity, no recompilation, exact frozen
trace and token-prefix parity. Report cross-policy token agreement, H2D and
peer bytes, cache hits, TPOT and controller CPU. A speedup is only credible
if primary TPOT improves without a material resource/correctness regression.

Use the standard supervisor's 384-GiB host preflight, 96-GiB stop and 85%
per-process HBM bound. Restore owned model loads after the job. Keep
`CA_NATIVE` opt-in until the evidence supports promotion; do not change Near.
