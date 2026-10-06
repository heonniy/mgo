# Owner-authorized TTFT execution

Authorization: 2026-10-06 chat explicitly requests execution of c68ecc2 TTFT,
then CA and OLD_CA both, and primary long-context ShareGPT (not MATH).

Primary scope is the prefill/TTFT track: C30/C60 x local B16/B128, physical
GPUs 0,1,4,5, exact512 valid prompt tokens, BR/CA/OLD_CA/LA, BF16, no
substitution or replication. Decode and joint-phase matrix remain separate.
TTFT measures a model-resident cold-expert-cache prefill through first-token
readiness, including live placement/admission and required H2D; model loading,
request delivery and compilation are outside the timed window.

The separately recorded long ShareGPT pool has 2048 distinct conversation prefixes
in corpus order with >=512 chat tokens, retaining the most recent512 tokens
per the owner amendment1804e65. Original lengths and conversation IDs/turns
are recorded. The unused first512 draft is preserved outside the repository.
No masked padding and no MATH requests. Capture the frozen prefill routes once
in bounded batches8 per assigned GPU, then reuse across all seed candidates.

S0 evaluates workload seeds0..31 and placement seeds0..31, retaining top8
per candidate and cache/batch. It is only a CPU proxy screen. All gain proxies
and physical comparisons pair candidate and BR on identical requests/seeds.
Audit seed activity: initial caches are empty; BR random admission is seeded,
while deterministic candidates must not be assigned artificial randomness.

S1: warm/correctness then one unprofiled paired timing per retained seed, with
order reversal across seeds. S2: fresh process on selected seeds, two paired
repeats; add one third only for2–5% TTFT difference, >5% is unstable. Keep all
valid samples. Selection samples are never the final gain estimate.

This is BEST-SEED HEADROOM, not an unbiased average-case comparison. Add an
opt-in prefill post-expert barrier with local-stream completion before return,
without mutating completed B5 measurement paths. Preserve exact routed work
and physical copy accounting and record cache-boundary hashes. Use dedicated
instrumented passes for phase breakdown, not primary timing.

Shared-server guards apply; only owned loads on0,1,4,5 are paused/restored.
Never touch2,3,6,7. Stop on OOM risk or correctness failure rather than changing
batch/context or reporting invalid timing. Commit checkpoints and failures.
