# Payload crossover — stopped at missing exact destinations

**Status: BLOCKED_MISSING_EXACT_DESTINATIONS.** The owner explicitly selected “계획대로 누락을 기록하고 중단”. Step B was not launched. No GPU microbenchmark, model generation, H2D test, new routing capture or replica implementation was performed.

## What was available

Four existing R4/B8 raw files were read deterministically: the C0 P0/P1/O0 captures and the earlier B8 oracle-planning capture. O0 and the planning capture are byte-identical, so these are three unique streams. Each file contains 3,120 layer events: 48 prefill events plus 3,072 decode events. Every decode event has raw selections of shape `[32, 8]` and exactly eight token origins per rank. Expert IDs and origin IDs passed range/shape checks. No trace was unreadable or corrupt in these checks.

## Required field that is missing

Exact ordered rank-pair activation sizes require the execution destination of each selected raw expert, in addition to expert IDs and token origins. The existing physical studies enabled substitution: some raw experts were not executed or resident because substitute experts handled their routes. Their substitute targets have destinations, but using those would violate the instruction to ignore substitution.

The audit checked the full post-event cache slots as well as inspecting execution metadata; it did not mistake a missing execution-list entry for a missing resident owner. The absent exact experts have no recorded destination in that event. A new CPU placement replay could supply hypothetical destinations, but would introduce a counterfactual placement convention; it would not recover an observed exact-only traffic distribution. No such convention was assumed.

| Existing source | Decode events | Events with missing exact owner | Missing raw expert/event pairs | Raw route occurrences without exact owner |
|---|---:|---:|---:|---:|
| b8_P0_C0-raw.pkl | 3,072 | 1,710 | 60,562 | 165,423 |
| b8_P1_C0-raw.pkl | 3,072 | 1,733 | 59,096 | 164,764 |
| b8_O0_C0-raw.pkl | 3,072 | 1,715 | 59,565 | 164,278 |
| plan_b8-raw.pkl | 3,072 | 1,715 | 59,565 | 164,278 |

For example, P1 event 48 (first decode, layer 0) selects raw experts 85 and 123, but neither has an exact execution/cache owner. The recorded substitution maps both to expert 43. Treating expert 43’s owner as their exact destination would produce a substituted-traffic result rather than the requested raw-exact result. Full first-missing examples and SHA-256 input hashes are in `payload_distribution.json`.

## Outputs and stopping point

`payload_distribution.csv` marks dispatch/combine as blocked and leaves all numerical distribution fields blank. `payload_distribution.json` retains the audit and uses null for unavailable quantiles. `payload_crossover.csv` intentionally has a header only. No new INFO logs exist because no transport smoke was run in this follow-up. These missing values are not zero-byte traffic measurements.

Runtime dimensions were verified from the existing checkpoint config: hidden size 2,048, BF16 two-byte elements, top-k eight and B8 per rank. These dimensions alone cannot determine the missing rank-pair destinations or quantiles.

Only the four requested compact result files are committed. The analysis script and audit remain at `/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/payload_crossover`; their provenance is included in the JSON. Stop for owner review; there is no new conclusion about the five-size transport crossover or the suitability of the H100 synthetic contrast.
