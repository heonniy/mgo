# BR-only physical tuning protocol

Owner updates: `OWNER_PARTIAL_RETURN_AMENDMENT.md` enables true rank-partial
return with reported numerical differences; use FP32 for the common main stack.
`OWNER_JITTER_AMENDMENT.md` supersedes the original repetition cap below with
bounded five/seven-sample jitter resolution. B256 reverses the B128 setting order
to reduce monotonic order bias. Keep BF16 V3 as a separate final candidate using
the same BR-frozen P/trigger.

The final comparison remains the frozen MATH LOAD workloads at decode256.
To avoid spending the overnight run on weak settings, screen the 18 BR-only
(B128/B256, P1/P2/P4, T0/T1/T2) combinations on the exact first64 decode prefix.
No new trace capture and no LA timing are used to select these settings.

Use the common validated runtime: compact metadata, live replicated controller,
priority CPU staging/H2D, ready-first compute, and exact two-A2A communicator
(M11 exact-return amendment). No primary measurement has per-event logging,
profiling or concurrent resource scans. Warm compilation and correctness checks
are outside measurement. Model generation retains native router cost and uses
common frozen routes, weights and teacher tokens.

Each cell has two clean MEASURE repeats. Apply the existing E2E/TPOT relative
difference rule: <=2% keeps two; >2% and <=5% adds exactly one third; >5% marks
unstable without additional timing repeats. Report every sample and full range.
Only settings stable for both batches are candidates. Rank shared settings by
geometric mean of BR TPOT normalized to each batch's best stable setting. Keep
the top two distinct settings, preferring a smaller P within 2% of the best.

Confirm these at most two finalists on the complete decode256 trace for both
batches, using the same repeat rule. Select a common P/trigger from stable full
runs by normalized geometric-mean BR TPOT, preferring the smaller P when both
batch TPOTs are within 2% of the best. Tie triggers in order T1, T2, T0. Freeze
before any final LA run. If prefix-screen settings are unstable or full runs
contradict the screen, diagnose the concrete cause; do not silently cherry-pick
samples or extend repetitions of an unstable pair.

Physical copies must reconcile exactly per rank:
mandatory + issued prefetch - canceled before submission = submitted copies.
MAIN state, role mappings and logical counters must match the independent CPU
replay. Frozen-input delivery may be staged before measurement, but future
routes are never passed to the predictor/controller.

Final arm selection remains maximum min(LA gain B128, LA gain B256). Before
seeing LA measurements, define material absolute-time domination as: another
stable arm is no slower in either batch and is >2% faster in at least one batch.
A dominated arm cannot win solely by making BR slower. Keep descriptive gains
for all arms, including unstable or rejected arms, clearly marked ineligible.
