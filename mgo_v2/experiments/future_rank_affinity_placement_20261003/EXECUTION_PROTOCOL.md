# Frozen execution conventions

- CPU only, one process, BLAS/OMP threads=1, CUDA hidden, hard 4-GiB
  address-space limit. Six policies per B8/B16/B32; no new GPU/model run.
- Owner instruction to stop GPU 2/3/6/7 supersedes the plan's stale wording
  about eight workers. Preserve running workers on 0/1/4/5 and do not restart
  stopped workers.
- All policies use F-identical prefill. Owner oracles apply to decode global
  misses only. Thus decode-start cache states match F; historical replicated
  K/C prefill states may differ. Horizons contain current decode plus the
  next 1/2/4 or remaining same-layer steps, truncated at step 8.
- Process misses in ascending expert ID. For current-event scoring, freeze
  resident owners and already chosen miss owners; undecided misses use F's
  max-current-demand owner. For future scoring, freeze other experts at
  independently validated F execution destinations. Change all routes of the
  candidate expert to the candidate single owner; include both remote dispatch
  union and combine rows. Costs omit only candidate-independent constants.
- Candidates have positive current demand; ties favor larger current demand,
  then lower rank. Do not add capacity/eviction penalties to the objective.
  Original atomic capacity preflight hard-stops any impossible chosen event.
- No optional fetch, duplicate or migration. Reuse the original replay's
  admissions, slots, active protection, LRU touches and exact traffic counters.
- Owner divergence compares each actual miss decision to max current demand
  at that same event, rather than comparing different policies' miss sets.
- Survival denominator includes decode oracle admissions with a later demand
  for the same layer/expert anywhere in the remaining trace. Count survival
  of that physical copy through the next demand; exclude right-censored
  admissions from that denominator and report them separately.
- Realized owner-attributable savings: for each served event, replace owners
  of still-resident changed-decision copies by the F heuristic owner recorded
  at their admission. Report joint exact traffic difference (interactions
  included). This controlled fixed-cache counterfactual is not the full F
  policy delta; cache-induced differences are captured by actual totals.
- Required deterministic repeated invocation: run each of the 18 fixed cells
  twice in separate sequential CPU invocations, comparing full event/state/
  decision hashes. These are correctness replays, not new policies, thresholds
  or statistical timing repetitions. Publish one coordinate per fixed cell.
- Verify raw source sizes/hashes and cross-rank receipts before each batch;
  independently check F at all 432 events, B8 totals exactly. Keep compressed
  event/owner/lifetime evidence outside Git with SHA256 receipts.
