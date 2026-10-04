# Owner amendment: rank-partial candidates are eligible for frozen-trace study

On 2026-10-05 the owner explicitly chose:
"수치·토큰 차이를 보고하고 포함 (권장)".

This supersedes M11_EXACT_RETURN_AMENDMENT's exclusion of rank-partial return
from performance candidates. Preserve and report numerical/token differences
instead of requiring bitwise equality with the legacy BF16 expert-order sum.
Frozen requests, selected expert IDs, routing weights and teacher tokens remain
identical across strategies. Cache state, expert contribution identity, balanced
quotas, physical copy accounting, no-replication and transport count correctness
remain hard gates. A change in reduction association must never hide a missing
expert, route, weight, copy or packet.

Implement and compare real coalesced forward and return: one destination partial
per received token/rank packet and exactly two payload A2As. Compare BF16, FP32
and FP64 accumulation diagnostically against the canonical high-precision sum
of the identical BF16 weighted expert contributions. Use short physical model
runs to report legacy-relative output/logit differences and argmax agreement.
Choose an explicit common accumulation format before resuming BR-only P/trigger
tuning. Report its wire-byte cost. Do not imply free-generation quality parity
from a frozen-route teacher-forced benchmark.

The prior exact-return runtime remains a valid diagnostic/ablation. Its first
B128/P1/T1 pair completed, with TPOT 1.600255 and 1.684813 seconds (5.148% spread),
and is ineligible as stable evidence. No extra repeat was added. Subsequent
P1/T0 untimed warmup was intentionally interrupted before further timing so the
coalesced candidate can be validated and incorporated into the common stack.
The driver's nonzero-exit receipt is an intentional replan, not a numerical or
resource failure. See M13_EXACT_SCREEN_INTERRUPTION.json. Do not resume the old
exact-only 18-cell queue automatically.
