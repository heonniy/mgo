# Large-cell first primary checkpoint

Repeat1 passes manifest coverage, raw-clock identities, exact64-token generation, GPU KV, cold parameter state and per-rank parameter cap checks. TTFT8.510366356s, TPOT5.064414795524s, E2E327.568498474s. Repeats2/3 remain in progress; no stability/headline claim.

This launch uses fixed nonoverlapping24-core guest CPU ranges. The raw affinity receipt wording predates the topology correction: all GPUs are exposed in NUMA0, so these ranges do not prove physical NUMA locality.
