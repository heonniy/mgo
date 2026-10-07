# CA versus main_OURS Near

The selected main_OURS remains native C++ expert execution, compiled
prefill/decode index paths, Near placement and prefetch OFF. This follow-up
tests CA as a policy candidate; it does not change that selection in advance.

Use R4/C30, local B16 (global64), input256, output64 on GPUs0/1/4/5 only.
Keep the full-pinned CPU expert source, 1,835 MAIN slots, two reserved and
unused physical prefetch slots per rank, BF16 rank partials and existing
overlap/barrier structure. Warm both policies, then capture one unrestricted
Near route and teacher-token stream. Reset to a cold expert cache for each
arm. Run Near/CA/CA/Near with 63 decode forwards and two uninstrumented
primaries per policy. Do not filter timing samples.

After the primary measurements, run one separately instrumented frozen
16-decode prefix per policy. Require each diagnostic's token prefix to match
its corresponding same-policy primary. Report TPOT, TTFT and E2E alongside
rank-level expert token rows, maximum expert-completion span, forward/return
collective spans, H2D bytes/misses/hit rate, peer bytes and controller time.
Do not treat instrumented spans as primary TPOT or pure network kernel time;
they include host submission and peer waits. Report cross-policy BF16 token
differences, finite logits, cache/role parity within a policy, and no
recompilation. Preserve all attempts and raw receipts.

Keep the supervisor's 384-GiB host preflight, 96-GiB stop and 85% per-process
HBM bound. Restore owned inference loads after the job. A CA promotion would
require a stable paired TPOT gain without a material correctness or resource
regression; otherwise retain Near as main_OURS.
