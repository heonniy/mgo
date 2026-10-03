# Frozen refresh protocol

Use only the d666414 B8/B32 captures, similarity file and validated replay.
No GPU process is stopped or launched. One CPU replay process, CUDA hidden,
BLAS/OMP one thread, hard 8-GiB address-space limit. Exactly 24 N0 cells are
validated first and reused in the 120-cell matrix (80 B32, 40 B8).

## Event order and eligibility

Preserve historical substitution, mandatory fetches and ordinary greedy
replication. After ordinary replication, before physical service/touches,
refresh only when the duplicate cap is full. A candidate is a currently
executing expert with current demand on a rank lacking its local copy. No
future-only prefetch candidate is introduced. A victim is an inactive key on
that target rank with at least two global copies. A primary copy is eligible
if another copy survives; historical primary promotion to the minimum
surviving rank is preserved. Every swap reuses the victim's physical slot,
preserves the exact global unique-key set and duplicate count, and incurs one
9-MiB replica fetch. C1/C2 accept at most 1/2 strictly positive current-byte
swaps; O4/OR accept at most one strictly positive horizon-byte swap.

Maximize the integer exact peer-byte reduction. Ties: candidate expert ID,
target rank, victim last-used event (oldest first), victim layer/expert key.
After each accepted swap recompute marginal savings. C1/C2 selection has no
access to future data. Ordinary evictions continue to use the selected LRU or
GATE policy. Refresh overrides the victim choice only for the authorized
inactive-duplicate swap.

## Oracle counterfactual

Score current service plus the next four (O4) or all remaining (OR) observed
same-layer **decode** events, strictly after the current global event. Apply
that horizon independently to both affected layers, taking their union; this
includes the victim's future locality cost when its layer differs from the
candidate's. Future prefill events are excluded; the current prefill event
is included. All refresh policies may act in prefill and decode.

Hold the present cache/primary maps fixed while scoring future demands. No
future eviction, replica admission, migration or gate-driven cache trajectory
is predicted. With substitution ON, call the unchanged production policy on
each future raw demand against the present global unique-resident set. The
before/after swap has the same unique set, so effective routes are identical
within each counterfactual pair. For a globally absent future effective expert,
use a virtual first-copy owner selected by the historical maximum current
rank-demand rule (lowest-rank tie), equally in both worlds; no virtual cache
mutation is applied. These virtual destinations are needed to count dispatch
coalescing with unaffected routes. Model routing itself is always frozen.

Count dispatch rows per distinct remote token/destination and combine rows per
remote effective expert. Include primary-promotion effects for all origins.
For same-layer swaps, joint dispatch interactions must be counted exactly;
addition gain minus removal loss alone is insufficient when both expert routes
share the candidate's old remote destination.

Future score caches are invalidated by physical owner changes and the moving
observation horizon. They cache calculations, not hypothetical cache evolution.
O4/OR are future-demand diagnostics, not globally optimal or guaranteed upper
bounds on a sequential placement controller's realized outcome.

## Accounting and interpretation

Use inherited full/decode counters and lifecycle definitions. Decode lifecycle
cohorts are decode-born replica admissions. A replica's admission-event service
does not count as reuse; promotion to primary does not end its physical life.
Observed lifetimes include right-censored lower bounds. Report the fraction
with no later physical service before eviction/end, with censoring counts.
Victim age reports both time since physical admission and time since last use,
in global layer events. Victims may have been first copies before duplication.

For each matched N0 comparison, record actual net peer savings and sum the
exact same-event savings measured at each accepted refresh. Their difference
is the **realized downstream peer-saving residual**: N0 peer minus the refresh
run's pre-refresh peer, summed over events. It includes subsequent admission,
eviction, primary and substitution trajectory effects of earlier refreshes.
It is not additive per-swap causal attribution, nor a frozen-oracle forecast.
Record it separately from predicted horizon gains and immediate savings.

Break-even byte prices are exact fractions for strictly opposite signed H2D
and peer deltas; identify which side of the price favors refresh. Dominance,
equality and same-direction deltas are labeled explicitly. J_byte is evaluated
only at the five declared prices. Labels use decode and fractional-rho matched
cells. ONLINE_REFRESH_PLAUSIBLE requires OR to have strictly positive peer
savings, avoiding zero/negative-denominator automatic passes. The H2D bound is
1.05 times OR H2D. CACHE_RELIEF_PRESERVED uses the same batch's committed
cache30/eviction/substitution/rho coordinate. NO_REFRESH_HEADROOM applies if
none of the current-only/oracle headroom labels passes, even if a cache-relief
context label passes. No accuracy, latency or hardware-slowdown claim.
