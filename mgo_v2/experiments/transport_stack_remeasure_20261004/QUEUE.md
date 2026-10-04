# Owner follow-up queue

Owner added 70780ff as a follow-up, not an interruption/replacement.
Order: resume c30/s1 (ac6d6bc protocol), then c60/s0 (70780ff protocol).
Each arm uses R/MATH/R4/B64/decode256 on GPUs0,1,4,5 with pinned H2D,
BR/CA and current/coslot/coslot-active. Env1 precedes conditional Env2.
Each condition has exactly three measurements; no extra repeats.
Cache and substitution identities, output directories and receipts are distinct.
The c60/s0 results are not a pure cache comparison against c30/s1, since both
cache ratio and substitution differ. This queue does not add c30/s0 automatically.

The c30/s1 BR PLAN generation succeeded, but a rejected push (remote branch
advanced) stopped the driver before CPU plan validation. The local phase and
handoff commits were merged with 70780ff without dropping either history.
Recover/validate that successful PLAN rather than generate it again.
A future push rejection records a pending-publication receipt and keeps the
local checkpoint; it no longer interrupts scientific execution.

CPU tests run using the same Python as the driver/worker; the system Python
has pytest but lacks Torch, while worker Python lacks pytest. A small direct
runner executes all seven existing test functions, including isolated temporary
paths and monkeypatch cleanup for the cache/substitution identity test.

Run scripts/queue_transport_cache_followup.py. Scientific failures stop the
queue for inspection. Completed or bounded unstable c30/s1 ends before c60/s0
begins. Both arms restore resident-model workers on exit; the next arm performs
its own exclusive GPU handoff. No OldCA or other prior queue is resumed.
