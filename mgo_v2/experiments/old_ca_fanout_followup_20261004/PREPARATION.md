# OldCA preparation checkpoint

The owner requested advance preparation. The predecessor subsequently completed
and committed as `ac65358` before candidate replay began. All eight resident
model workers were restored by that predecessor; this preparation is CPU-only.

Historical a0be82e costs and SciPy Hungarian assignments match in 120 fixtures.
Explicit tests cover pre-owned destination coalescing, immutable admission
base, balanced quotas, BR/CA regression, and batch/incremental cache parity.
Policy 3 is opt-in, substitution-off, with no affinity bonus.

Reuse all existing BR/CA receipts. Replay OldCA once on 248 retained candidates;
one original candidate is safely pruned because Current-CA already exceeds
0.25% for every BR seed. No seed expansion or new trace capture occurred.
CPU workers use the existing adaptive, single-thread, memory-guarded scheduler.

| R | Sample / DP / BR seed | Common maximum H2D difference | OldCA fan-out reduction vs BR |
|---|---|---:|---:|
| 4 | 251 / 42 / 19 | 0.09925% | 28.8604% |
| 8 | 484 / 13 / 73 | 0.09634% | 24.3207% |

Both qualify at the preferred 0.1% bound; fallback is unnecessary.
These are CPU communication results, not measured speedups.
See winners.json, selection_validation.json and CPU_receipts.json.

The background plan builder creates all three policies with identical requests,
teacher tokens, expert IDs and weights, and checks physical-plan fetch/peer/
critical totals and cache hashes against the CPU receipts. Gzip level 1 reduces
preparation time without changing decoded schedules. Raw logs and process IDs:
`/home/hwlee/mgo-results/old_ca_fanout_followup_20261004/`.

GPU timing has not started. The remaining physical integration must use A3 only,
R4 then R8, counterbalanced three-policy order, and the confirmation/reuse rules
in PLAN.md. The plan builder does not itself launch GPU jobs.
