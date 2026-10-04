# PLAN — R8/B128/c60 high-cache/high-batch stress

## 1. Question

At R8, local B128 and 60% global cache, can:
1. communication-aware one-copy placement, or
2. hot-expert replication

materially reduce the bottleneck relative to an intentionally bad random
one-copy placement?

This is a **best-headroom stress experiment**. It is not an average-workload
evaluation.

## 2. Why this point

R8/B64/c60 exact-only already showed high global cache hit but low local hit.
B128 doubles per-rank request parallelism and should increase per-expert token
rows, making expert-compute imbalance a more plausible bottleneck.

At c60 the first-copy/coverage value is lower than at c30, so extra copies have
more opportunity to become useful.

## 3. Reused data

Reuse both exact 2048-request candidate pools from the completed CA stress
search. R8*B128 consumes 1024 requests, so there is still subset-selection
freedom. Do not capture new GPU traces.

## 4. Search variables

- sample_seed = 0..255;
- dp_seed = 0..255;
- BR placement_seed = 0..255.

The three variables have different roles:
- sample seed changes global hot-expert demand;
- placement seed determines which one-copy owner ranks receive those experts;
- DP seed mainly changes remote/local activation demand and can also influence
  Gate-history ordering.

## 5. Two-stage bounded search

### Stage A: route-only structural prescreen

For each dataset and sample seed:
- use 48 frozen proxy events;
- compute global per-expert row counts;
- estimate worst balanced one-copy rank-load concentration.

Retain 16 sample seeds per dataset.

For each retained sample, search placement seeds 0..255 with a balanced
one-copy random owner proxy. Retain the eight strongest sample/placement pairs.

For each retained pair, search dp_seed 0..255 using remote-route/token-rank
proxy cost. Keep eight triples per dataset.

No CA or replica result may affect this selection.

### Stage B: exact Gate replay

Exact-replay the 16 retained triples over all 256 decode steps using real
BR one-copy Gate cache semantics. Also replay deterministic CA on the same
sample/DP pair.

Select one final winner using BR-only exact metrics:
1. decode sum(max-rank expert rows);
2. decode peer bytes;
3. decode peak max-rank expert rows;
4. deterministic seed tie order.

Freeze request IDs, per-rank assignment, route/gate hashes, BR seed and final
cache hashes.

## 6. Replica oracle

For every exact replay event, evaluate one temporary replica.

Variant A — local-origin-shift:
for expert e owned by p and candidate replica rank q, move only e's rows whose
origin is q from p to q. This represents the most conservative local replica
use.

Variant B — free-split-compute upper bound:
split e's rows between p and q to minimize the event's maximum rank expert-row
load, irrespective of origin. This isolates pure compute-balance headroom.

Choose the best (expert, q) per event. The cache state itself is not changed.
Thus H2D/cache capacity remain those of the one-copy replay and the oracle is
not a realizable performance result.

Required oracle metrics:
- sum and peak max-rank rows before/after;
- mean event rank-load CV;
- fraction of events improved;
- chosen hot-expert demand distribution;
- ideal fully divisible lower bound ceil(total expert rows / 8).

## 7. Comparison

For the frozen winner report:

| Policy | Critical rank rows | Peer bytes | H2D | Global hit |
|---|---:|---:|---:|---:|
| BR one-copy | measured | measured | measured | measured |
| CA one-copy | measured | measured | measured | measured |
| BR + 1rep oracle | upper bound | BR cache | BR cache | BR cache |
| CA + 1rep oracle | upper bound | CA cache | CA cache | CA cache |

Also report CA's peer reduction separately from replication's compute-load
headroom; do not collapse them into an arbitrary weighted score.

## 8. Go/no-go

Proceed to a real same-capacity persistent replica implementation only if
either BR or CA temporary-replica oracle reduces decode
`sum(max_rank_expert_rows)` by >=10%.

If neither reaches 10%, stop replication work for this regime.

## 9. Physical timing

None in this packet. Physical E2E/TPOT follows only after the oracle gate and
requires an implementable cache-capacity-preserving replica schedule.
