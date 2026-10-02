# Decision-equivalent controller overhead study

Authorization: `rank_demand_oracle_20261002/PLAN.md` section 12, added at
`a6a4a85` and clarified at `f2ebba5`. This document fixes the follow-up measurement
rules before implementing or measuring either optimized controller.

**User scope reduction, 2026-10-02:** the owner subsequently requested the
minimum sufficient experiment set. This amendment overrides the larger timing
matrix and expansion gate below: run B8 only, one primary generation per
policy/controller (9 generations total). Run fresh C0 controls first, then C1,
then C2 after parity gates. Full captured-route CPU differential replays supply
component diagnostics. After timing, use short P1 GPU profiles for C1/C2 to
check device work and measure the added planner transport. Keep complete event/cache/token parity and separate
diagnostics. Do not expand to B4/B16. Treat performance as single-sample descriptive evidence, with no
repeatability claim, and report the resulting order limitation. See
`../rank_demand_oracle_20261002/scope_amendment.json` for the exact authorization
and reduced placement matrix. The earlier rules below remain as audit history.

## Stage boundary

Finish and commit the original 54-generation oracle packet and its three B8
profiles first. Its controller, runtime, worker and launcher remain unchanged.
Store all controller-stage controls, traces and results separately. Fresh C0
controls belong here; they never replace the original placement comparison.

## Variants and invariants

- C0: exact original controller implementation.
- C1: opt-in local optimization of Coverage ranking, resident/cache views and
  validation-only full scans. Keep admission, substitution, history arithmetic,
  Coverage score, victim tie-breaks, pinned owners and cache semantics exact.
- C2: C1 on planner rank 0; a compact integer decision payload is broadcast to
  followers, which apply that plan to their logical cache and execute it. All
  ranks still contribute the same global routing metadata. Measure payload
  bytes and collective/copy/decode costs explicitly.

Preserve the original checkpoint, workload, physical GPUs 0/1/4/5, NUMA binding,
BF16 expert order, cache .30, W128, Coverage k=1/lambda=2, substitution .20/.65,
seed 42 and balanced admission-count quotas. No replication or migration.
P0/P1 still compute their unchanged online admission decisions; O0 uses the
original frozen oracle records. Every variant must reproduce its policy's
original routes, decisions, victims, full cache state and generated tokens.

## Correctness and diagnostics

Before optimized timing claims, run existing CPU tests plus C0/C1/C2
event-level differential tests. Compare the complete substitution and admission
objects, execution operations, effective-route weights, owner/slot maps,
timestamps and rolling history; run full cache/index consistency checks
outside primary timing. Test payload round-trip and fail-closed rejection.

Begin with full R4/B8 diagnostic generations for all nine policy/controller
combinations. C1 parity must pass before C2 measurements. Detailed CPU timers
and GPU profiling belong only to these diagnostics. Retain per-event/rank
controller wall time, exclusive Coverage/admission/substitution/history/cache
components, candidate visits and list/set materializations. Count C2 transport
separately from unchanged expert fetch traffic. Match physical expert rows,
GEMM counts, fetch bytes and outputs; device durations may change with host
scheduling and are measured, not required to be equal.

## Primary timing order and expansion gate

Use one prefill plus 64 decode forwards, with fresh logical and physical cache
and history per cell. Hashing and serialization occur outside the generation
timer. No subcomponent timers, CUDA events or profiler in primary timing.

At B8, use all six permutations of (P0, P1, O0) once. Within each policy position,
run C0/C1/C2 in one of all six controller permutations, rotated by policy index.
Thus every policy/controller combination has six repetitions, each policy sees
all six controller orders, and B8 contains 54 primary generations. Publish the
exact order manifest before starting measurements. Summarize medians and full
observed ranges, without presenting the range as a confidence interval.

Predeclared material-reduction gate: at least one optimized variant must reduce
the median maximum-rank total controller time by at least 10% versus fresh C0
for **each of P0, P1 and O0**, with all event/cache/token parity checks passing.
Only then run the same 54-generation primary matrix at B4 and B16. Otherwise
stop the expansion and report the B8 outcome. Do not retune policy coefficients
or relax parity in response to timings.

## Resource controls and reporting

Run only one four-rank job at a time. Reuse the original memory guards: launch
with at least 512 GiB host availability and selected GPUs effectively idle;
stop this stage's own process tree if host availability falls below 128 GiB,
tree RSS exceeds 320 GiB or a selected GPU has less than 8 GiB free. Do not
touch processes on GPUs 2/3/6/7. Stream profile analysis with bounded memory.

Report controller reduction separately from policy-placement differences,
together with TPOT/E2E, controller fraction, Coverage costs, candidate visits,
C2 broadcast bytes/time, GPU work and shared-host limitations. Preserve raw
logs externally and commit the separate report, compact receipts, source
hashes, validation gates and figure-support CSVs.
