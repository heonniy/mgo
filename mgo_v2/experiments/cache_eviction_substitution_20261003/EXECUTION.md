# Frozen execution and accounting

The owner-authorized 264-cell matrix is unchanged. S0 makes exactly B8 and
B32 feature captures; tokens, raw routes, selected weights, owner maps,
plan/cache hashes must equal the corresponding historical receipt. Each event
adds only its 128 float32 W128 gate scores. No router tensor is retained.
All four ranks must have equal gate-vector/plan/cache hashes. GPU 2/3/6/7
are excluded from worker handoff. The original captures and similarity file
are SHA256-pinned in trace_readiness.json.

CPU replay uses the existing SubstitutionPolicy and merge_effective_routes.
Gate vectors replace only the current layer's scores; untouched layers retain
the previous observed history (initially zero). Routing remains frozen exact
model demand: this is not a substituted autoregressive model execution.

LRU preserves the historical replica replay's `(last_used, key)` ordering.
GATE uses `(gate score, last_used, key)`. COVERAGE uses production doubled
integer percentile midranks for gate + 2*unique-coverage damage, followed by
last_used/key. A nonlast copy has zero unique damage. Coverage includes all
unique residents, including remote and pinned experts. Each event pins all
merged effective experts. The production substitution thresholds and tier
ordering are unchanged. Source counts count source expert-events; similarities
are weighted once per accepted source expert-event, not by token multiplicity.

The historical greedy marginal is evaluated with integer route counts. It
counts one saved combine row per moved expert route and a saved dispatch row
only when that token has no remaining expert on its previous remote rank.
Ties remain expert ID then rank. Physical entries served in the current event
are the only entries touched. A small independent differential test checks
all action/state/counter transitions; the five full historical B8 replays must
also match every common full/decode counter and final state hash.

Each replica admission starts a physical-copy lifetime, ending at its eviction
or at the end of event 431 (censoring time 432). A replica can later become
the primary; its physical lifetime continues. Reuse requires service in a
strictly later event; service during its admission event is excluded. Full
and decode lifetime summaries select admissions born in the corresponding
window. Thus prefill-born copies serving decode affect decode traffic but
are not decode-born admissions. Eviction counts instead count all evictions
occurring within each window. Lifetimes are measured in global layer events.

Reported lifetime p50/p90 use observed durations, including right-censored
lower bounds; they are not Kaplan-Meier estimates. Survival>=48 fraction is
observed survivors divided by all admissions (a lower bound); short censored
lifetimes are explicitly counted as unknown. A known-outcome fraction is also
reported. CACHE_RELIEF uses the all-admission lower bound. If cache30 survival
is zero, a strictly positive larger-cache survival meets the 2x clause;
zero versus zero does not. An empty replica cohort has null survival and
does not meet a survival clause. Reuse among evicted copies and reuse observed
among all admitted copies are reported separately.

Lower envelopes use exact rational byte prices on decode and full separately.
`lambda_first` is the earliest nonnegative price interval with a nonzero-rho
winner: zero when replication is already optimal at zero; null if never.
Historical B8 cache30/LRU/OFF must recover 7435008/17693=420.223139094557.
The interpretation labels use decode metrics. EVICTION and SUBSTITUTION
comparisons include fractional-rho grid cells only. CACHE_RELIEF uses B8
fixed-460 controls, matched eviction/substitution. The replication dominance
clause compares B8 nonzero-rho cells against the historical B8 cache30/LRU/OFF
cell at the same rho. No historical B32 frontier is invented. B32 may meet
the explicitly prescribed lambda threshold against the numerical historical
anchor; that cross-batch comparison is labeled as such. Pareto dominance
requires at least one strict byte reduction and no increase on the other axis.

The five S1 validation runs are the five corresponding B8 grid cells and are
not repeated. Fixed-460 controls are separate matrix cells, including cache30
where they provide a direct equality check against rho=.25. One CPU process,
CUDA hidden, BLAS/OMP one thread, hard address-space limit 8 GiB. Compact cell
checkpoints and progress preserve completed work without automatic reruns.
