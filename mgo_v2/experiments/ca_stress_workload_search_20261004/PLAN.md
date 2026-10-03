# PLAN — CA communication-stress search

Date: 2026-10-04
Status: owner-authorized follow-up after timing diagnosis.

## 1. Question

For each `dataset x R x local-B x cache-ratio` cell, which real request subset
and balanced DP rank assignment create the largest realizable communication
advantage for CA over BR?

The target is not "average performance". It is a reproducible upper/stress
regime for the placement mechanism.

## 2. CA/BR invariant

For an event with `m` currently missing unique experts and `R` ranks, both BR
and CA use the same quota multiset:

```
floor(m/R) or ceil(m/R) mandatory admissions per rank
```

Thus for every event:

```
max_rank_fetches - min_rank_fetches <= 1
```

BR randomly maps miss experts onto those balanced rank slots. CA uses the same
slots but solves the Hungarian assignment that maximizes current rank-local
exact expert-route demand.

Important limitation: equal quota is per current event. BR and CA can later
have different cache states, so their future miss sets and cumulative H2D need
not be identical.

## 3. Frozen scientific axes

Datasets:
- MATH;
- ShareGPT V3 cleaned.

Main decode horizon:
- 256 generated tokens.

System:
- R in {4, 8};
- local B in {8, 16, 32, 64};
- global requests N = R * B;
- global cache ratio in {30, 40, 50, 60}%;
- Gate eviction, W=128;
- substitution OFF;
- exact routing only.

Policies:
- BR;
- CA.

No LRU, substitution, CA-rep, replica refresh, or policy timing is part of this
search.

## 4. Candidate request pool

The existing 512-request exact-model traces are sufficient for initial search
and must be reused.

Because R8/B64 consumes all 512 existing requests and therefore has no subset
selection freedom, extend each dataset to a **2048-request candidate pool** if
the dataset/filter provides enough eligible requests.

Requirements:
- preserve the original 512 requests as a prefix/subset;
- same Qwen3-30B-A3B-Instruct-2507 checkpoint and prompt formatting;
- exact model, no substitution/intervention;
- one prefill + 256 decode route steps;
- use all 8 H100s in parallel for capture;
- capture only the additional requests needed to reach 2048, not the original
  512 again;
- this is correctness/route capture, not a timing experiment;
- hash-pin dataset IDs, prompts, token outputs and route artifacts.

If a dataset has fewer than 2048 eligible requests under the existing filter,
use all eligible requests and record the pool size. Do not silently alter the
filter to reach 2048.

## 5. Search variables

For every R/B group:

1. `sample_seed`: reproducibly shuffle the candidate pool and take first N.
2. `dp_seed`: reproducibly shuffle those N requests and assign exactly B
   requests to each rank.
3. `br_seed`: BR's random mapping of miss experts to its balanced quota slots.

The CA assignment is deterministic given the request/rank demand and cache
state.

### Primary stress candidate

Do **not** select a request subset only because one unlucky BR seed is bad.
Choose the sample/dp pair using a structural score robust across BR seeds.

Use predeclared BR seeds:
```
[7, 19, 42, 73, 99, 131, 181, 251]
```

For an exact replay define:
```
gain_rel = (peer_BR - peer_CA) / peer_BR
gain_abs = peer_BR - peer_CA
```

Primary score for sample/dp selection:
1. maximize median `gain_abs` across the eight BR seeds;
2. tie-break by median `gain_rel`;
3. tie-break by lower CA peer bytes.

After the primary sample/dp pair is frozen, also report:
- the median over the eight BR seeds;
- the best observed BR seed among the same eight seeds;
- the full min/max seed range.

This gives both a defensible structural stress case and the requested
best-observed random seed without hiding seed sensitivity.

## 6. Two-stage search

### Stage A — cheap routing-only prescreen

For each dataset/R/B:
- sample_seed = 0..255;
- dp_seed = 0..255.

Use raw exact routes to estimate rank-demand locality and the instantaneous
balanced-assignment headroom. No cache simulation and no timing.

Retain the best 32 distinct sample/dp pairs by proxy score, while requiring
exactly B requests/rank.

### Stage B — exact cache replay

For the retained candidates, replay the exact Gate cache semantics for every
cache ratio {30,40,50,60} and the eight BR seeds above.

For each `dataset x R x B x cache`, freeze one primary sample/dp pair using
the score in Section 5.

Because cache ratio changes the cache trajectory, the selected pair is allowed
to differ across cache ratios.

CPU/Numba should perform this combinatorial replay. GPUs are not an efficient
place to run the seed search itself.

## 7. Outputs

Produce one row for every:
```
dataset x R x B x cache
```

Required columns:
- candidate pool size;
- global N;
- sample_seed;
- dp_seed;
- selected request IDs / manifest hash;
- rank-to-request manifest hash;
- primary median BR seed gain_abs / gain_rel;
- best observed br_seed and gain_abs / gain_rel;
- BR peer bytes;
- CA peer bytes;
- BR H2D bytes;
- CA H2D bytes;
- exact global/local hit rates;
- per-event quota validation;
- final cache-state hashes.

Also report a compact table of the top-3 candidates per cell so the chosen
stress point is not opaque.

## 8. GPU use

Use all 8 H100s only for:
- extending the exact-model candidate trace pool to 2048 per dataset;
- optional correctness re-capture if an artifact fails validation.

Do not use GPU E2E timing in this packet.

After trace capture, run the large seed/subset search on CPU while following
the repository CPU resource guards. If the timing diagnostic failed, this
offline search still proceeds. If it passed, do **not** automatically start a
large physical timing sweep; commit the stress-search results first.

## 9. Interpretation

The final rows are intentionally optimized communication-stress examples.
They may be used to show mechanism headroom, but should be labeled as such.

For a paper-quality presentation, keep at least one neutral/random workload
result next to the optimized stress result. Do not present the optimized row as
the dataset-average behavior.

## 10. Stop rule

Complete the full resource search, commit manifests/results/validation, and
stop for owner review. No automatic BR/CA physical timing matrix follows.
