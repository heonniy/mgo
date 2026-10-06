# Large-cell first primary checkpoint

Repeat1 passes manifest coverage, raw-clock identities, exact64-token generation, GPU KV, cold parameter state and per-rank parameter cap checks. TTFT8.510366356s, TPOT5.064414795524s, E2E327.568498474s. Repeats2/3 remain in progress; no stability/headline claim.

This launch uses fixed nonoverlapping24-core guest CPU ranges. The raw affinity receipt wording predates the topology correction: all GPUs are exposed in NUMA0, so these ranges do not prove physical NUMA locality.

Repeat2 passes the same finite-output, GPU KV, cold-state, token-count and parameter-cap guards. TTFT5.795124850s, TPOT4.728005424381s, E2E303.659466586s. Two-sample relative differences (absolute difference / mean): TTFT 37.961%, TPOT 6.871%, E2E 7.575%. Repeat3 is running. Fixed CPU ranges did not eliminate this observed variation; no NUMA or scheduling cause is established.

All3 repeats completed; full auditor passes correctness/resource/manifest/raw-clock guards but marks timing UNSTABLE. TTFT: median 5.795124850s, full range [5.694123932, 8.510366356]s, spread 42.244%; TPOT: median 4.728005424s, full range [4.693160584, 5.064414796]s, spread 7.689%; E2E: median 303.659466586s, full range [301.363240696, 327.568498474]s, spread 8.430%. All samples retained; bounded confirmation remains required before headline selection.
