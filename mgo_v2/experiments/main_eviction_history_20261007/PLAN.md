# MAIN eviction history comparison

R4/C30, local B8/B16/B64, input256. Empty cache/history before prefill;
retain prefill state for exactly256 decode forwards (257 output tokens).
One exact live source trace per batch, reused unchanged by all five CPU policies.
Prefetch OFF, no substitution/replication; all1843 C30 slots are MAIN
(461/461/461/460). BR admission algorithm and seed42 are held fixed, not owner
maps: residency and misses differ between policies. Source routing may depend
on its BF16 arithmetic; results are conditional on this frozen source trace.

Policies: gate-score W128; LFU-reset; LFU-cumulative; LRU-reset; LRU-cumulative.
LFU increments ONCE per distinct expert per layer/step, independent of token
count. Reset deletes policy frequency/recency on eviction; cumulative retains
it across evictions and rank re-admissions. All histories are initially empty.
LRU latest use is updated on admission/access, so reset and cumulative must
produce identical cache decisions. LFU ties use last-use then expert key;
gate ties use last-use then expert key. Current active experts are protected.
Analytical ever-admitted/reload counters survive eviction for every policy;
they are never used in eviction decisions. Reload means a miss for a previously
admitted expert, irrespective of which rank previously held it.

Primary counts: pre-admission MAIN distinct-expert hit/miss, actual eviction,
and reload. One request per (decode step, layer, expert); prefill is separate.
Report per-step and cumulative totals plus checkpoints1/8/16/32/64/128/256.
Gate source-versus-replay count and final-state parity, analytical traces for
LFU reset/cumulative divergence, LRU equivalence, and conservation checks are
required. Missing/short traces fail closed. This is a cache-policy comparison,
not a timing or quality claim. GPU captures only0/1/4/5 serially under384/96GiB
guards. Five policies replay on CPU; no GPU run per policy.
