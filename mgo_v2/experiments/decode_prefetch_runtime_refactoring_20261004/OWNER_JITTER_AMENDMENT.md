# Owner amendment: investigate jitter and add bounded repetitions

The owner explicitly authorized adaptive extra repeats when measurements are
noisy, rather than stopping at the earlier two/three-sample cap. This supersedes
the repeat-limit paragraphs in earlier protocol documents; all observations
remain visible. Do not delete an inconvenient sample or repeat until a favorable
number appears.

## BR-only configuration screen

Start with two clean samples. <=2% E2E/TPOT spread keeps two; >2% and <=5% adds
one third. A >5% spread, including after the third, extends to five total. At
five, require both E2E and TPOT mean 95% Student-t confidence-interval half-width
<=2% of the mean, and first-two versus last-two mean drift <=5%. If unresolved,
add exactly two more samples, for at most seven. Unresolved estimates at seven
are ineligible and trigger diagnosis, not another unbounded repetition loop.
Report mean, median, full range and all observations. Use the mean for the
five/seven-sample estimate; the original stable three-sample median is retained.
These small-sample intervals assume approximately stationary timing noise and
are not a guarantee against external host changes.

## Final BR-versus-LA comparisons

Alternate order by round: BR/LA, LA/BR, BR/LA, etc. Both policies complete each
round. Apply the initial two/three rule; paired gain variation >2 percentage
points or noisy absolute timings trigger five complete pairs. At five, evaluate
the mean paired log(BR/LA) ratio and transform its 95% Student-t interval back to
gain. Require E2E and TPOT gain interval half-width <=2 percentage points and
first-two versus last-two paired-gain drift <=2 points. Otherwise extend to at
most seven pairs. Keep every valid pair. A positive gain claim requires the
interval's lower bound above zero; a precise interval crossing zero is reported
as no demonstrated improvement.

A stable paired gain can coexist with common-mode absolute-time jitter. In that
case explicitly report that absolute E2E/TPOT still vary; do not relabel their
full ranges as stable. If relative estimates remain unresolved at seven, do not
use that arm as a winner. Diagnose a concrete runtime or environment issue and
preserve the previous cohort if a repair justifies a new version/cohort.

## Nonintrusive diagnostics

Record GPU clocks/power, host load, available memory and temperature only at
measurement boundaries. Record process CPU time and context-switch deltas before
and after generation; no periodic process/GPU scan, profiling or per-event log
runs inside clean MEASURE regions. No shared GPU clock/power settings are
changed. Separate untimed profiles are used to investigate a suspected cause.

The earlier exact-return P1/T1 pair remains recorded as unstable under its
original rule. Its queue was intentionally replaced by the owner-authorized
coalesced-return study; do not silently reclassify its two samples as conclusive.

Fresh measurement workers also take two own-thread CPU counter snapshots,
before and after generation, outside its E2E/TPOT timers. Main-thread and
expert-staging-thread CPU time can distinguish controller/launch activity from
staging activity; aggregate process CPU counters remain available as well.
This is not continuous sampling. The current host has kernel scheduler wait
statistics disabled, so queue-wait values are explicitly null and no system
setting is changed. Snapshot scope includes generation entry/exit overhead;
do not interpret it as a precise decomposition of TPOT.
