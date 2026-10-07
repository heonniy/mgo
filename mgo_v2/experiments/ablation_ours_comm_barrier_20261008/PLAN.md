# ablation_OURS: rank-synchronized token collectives

Keep the selected `main_OURS` unchanged: R4/C30, GPUs 0/1/4/5, local B16,
input 256, output 64, full-pinned CPU experts, native C++ expert executor,
compiled prefill/decode layout, BF16 rank partials and prefetch OFF.

The opt-in `MGO_NATIVE_SYNC_ABLATION=1` changes decode timing only. For each
layer, the forward token packet is fully materialized and all required local
H2D copies complete before a local CUDA-stream completion and an all-rank
barrier. The dispatch all-to-all is then submitted. After native expert work
and return rank-partial packet construction, another local completion and
all-rank barrier precede the return all-to-all. Both barriers are immediately
adjacent to the actual `all_to_all_single` calls. Count exactly one of each
per decode layer; no barrier is added to prefill. The normal mode follows the
unchanged transport fast path.

Capture one unrestricted Near route and teacher-token stream for each job;
verify matching trace SHA256 across the normal and ablated jobs. For both
modes, run cold-cache BR/CA/CA/BR, two uninstrumented primary timings each,
using that job's frozen route. Then run a separate instrumented 16-decode
prefix per policy to partition barrier wait, collective submission/completion,
H2D, expert work and controller time. Compare BR/CA TPOT and peer bytes as
primary evidence. The diagnostic post-barrier collective span includes NCCL
launch and completion, so label it communication-path cost rather than
literal wire-only latency. Do not use instrumented timings as primary TPOT.

Require no recompilation, finite logits, token-prefix parity, deterministic
same-policy cache/controller/byte receipts, and the same native kernel in both
modes. Preserve raw attempts. Use the supervisor's host/HBM guards and restore
owned model loads after each job. Do not promote this mode into main_OURS.
