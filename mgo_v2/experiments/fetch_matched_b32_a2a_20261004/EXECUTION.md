# Immediate decode64 replacement

The owner cancelled the remaining old R4 experiments and queue. The partial
R4_0146 BR/Env1 first MEASURE is invalidated; R4_0123 never starts. Completed
R8 results are retained. This overrides the earlier wait-for-old-R4 wording.

No new trace capture. Use ShareGPT's audited 2048-request pool and exactly
its first64 decode events, plus the original prefill. Generated tokens are
sliced [:,:64]. Selection/router/Gate inputs never use later decode events.

Stage A freezes 48 routing-only proxy events: layer0..47 once, decode index
(layer%4)*21, hence 0/21/42/63. For each sample0..511 and DP0..63, use the
existing NumPy seed-permutation convention with exactly32 requests/rank.
Compare a feasible balanced greedy-locality assignment with balanced BR42
on these cache-free events. BR42 here is only a fixed proxy reference, not
selection of the exact BR seed. Compute both remote activation+return peer
bytes and sum of per-collective max-rank(send+recv) bytes. Retain independent
top64 rankings, deterministic sample/DP tie order; replay their union exactly.
This is an approximate prescreen, not a claim of globally optimal selection.

Exact replay preserves the original Gate64 cache policy, adding remote
activation/return per-event rank traffic counters only. The full returned
policy arrays must match the uninstrumented reference in a real64-prefix
compatibility gate. Traffic excludes self transfers. Dispatch and return
are separate collective events; sum their individual rank maxima rather
than taking one maximum of whole-run traffic. Weight bytes are reported
separately (16 bytes/dispatched token-rank versus4096 activation bytes).

CA runs once per retained candidate; BR uses exactly the eight planned seeds.
Require integer total-fetch and H2D equality, no tolerance or rounding. Apply
owner tie-breaks, then deterministic ascending sample/DP/BR seed for any
remaining tie. Save both objective winners, top10 and all eligibility counts.
No expanded search is allowed if exact matches are absent.

CPU execution reuses shared read-only memory maps and the adaptive single-thread
worker manager: begin8, increase with useful throughput, reserve16 vCPUs and
320GiB, cap8GiB private data/96GiB address space per worker; abort below256GiB
host available. GPU resident-model workers may run during CPU search and are
stopped before any scientific GPU work. Large artifacts stay outside Git.

Physical stages must follow PLAN.md's common frozen-route teacher forcing,
A2 correctness gate, Env2-only R4 then R8, short one-time readiness and bounded
single-shot/one-confirmation rules. No old 2/3 repetition rule is carried into
this exploratory screen. Commit each stage and restore eight resident models
when the packet finishes. No automatic extension.
