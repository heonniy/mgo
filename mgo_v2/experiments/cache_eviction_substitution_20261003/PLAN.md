# PLAN — Cache relief, eviction policy, and substitution

Date: 2026-10-03
Status: prospective, owner-authorized system headroom study.

## Motivation

All previous F/K/C and selective-replication results used **cache ratio 30%,
LRU eviction, exact-only routing**. At that setting:

- total global slots = 1,843 = floor(48*128*.30);
- per-rank slots = [461,461,461,460];
- B8 F decode = 154.995 GiB H2D / 319.453 MiB peer;
- B8 K (rho=.25) = 219.076 GiB H2D / 185.297 MiB peer;
- selective replicas often died before the next same-layer decode opportunity.

The completed reuse study showed substantial repeated rank-local demand, but
replicas were usually evicted before useful reuse. This packet asks:

1. Does a larger cache reduce the unique-coverage penalty of replicas?
2. Does a non-LRU victim policy keep useful experts/replicas alive longer?
3. Does cache-aware substitution lower miss pressure enough to reshape the
   fetch-vs-communication frontier?

This is a **system accounting** study. Substitution quality/accuracy is not
measured here and no accuracy claim is allowed.

---

## Frozen model/policy parameters

R4 Qwen3-30B-A3B-Instruct-2507, 48 MoE layers, 128 experts/layer, top-k 8.

Cache ratios:

```
{0.30, 0.40, 0.50, 0.60}
```

Balanced physical slots use floor(6144 * cache_ratio), divided across ranks.

Eviction policies:

### LRU
Existing physical-copy LRU, active experts protected.

### GATE
Evict lowest recent router importance, then LRU tie-break. Use the existing
gate-history convention W=128.

### COVERAGE
Existing gate+similarity diversity objective:
- gate history W=128;
- similarity threshold=.65;
- coverage k=1;
- lambda=2;
- LRU tie-break.

For a duplicated expert, evicting one copy has zero unique-coverage damage if
another copy survives; only loss of the final global copy contributes coverage
damage.

Substitution OFF/ON:

ON uses the current validated policy unchanged:
- expert-level decision;
- gate protect threshold=.20;
- similarity threshold=.65;
- tier1 anchors = exact hits + protected exact misses;
- then inactive resident safe anchors;
- residual misses stay exact.

Do not tune thresholds.

Replica budgets:

```
rho={0,.125,.25,.5,.75}
```

rho is the maximum duplicate-slot fraction of the current cache capacity.

---

## S0 — policy-feature trace readiness

Existing batch captures contain selected experts and selected routing weights,
but the production GATE/COVERAGE policies use a W=128 history derived from full
router probabilities.

Create exactly two short **diagnostic exact-only captures** if needed:
- local B8 / global B32;
- local B32 / global B128.

Use the same frozen prompts and exact model path as the existing validated
captures. One prefill + eight decode forwards. No timing claim.

Do **not** store the full router probability tensor. Instead, after each layer
updates GateHistory, store only the resulting 128-element float32 gate-score
vector for that layer/event. Also retain:
- raw selected experts;
- selected routing weights;
- origin ranks;
- exact owner/cache hashes;
- generated-token hash.

Require generated tokens and raw selected routes to match the corresponding
existing exact capture. Hash the existing similarity.npy used by mgo_v2.

If the existing raw receipts already contain sufficient exact gate-history
state, reuse them and skip recapture.

Maximum new model captures: 2.

---

## S1 — replay validation

Implement one CPU replay supporting:
- variable cache capacity;
- LRU/GATE/COVERAGE physical-copy eviction;
- optional substitution;
- optional duplicates.

The replay order for every event is frozen:

1. update/read the event's frozen gate-history scores;
2. compute substitution from pre-event global residency;
3. merge effective routes;
4. admit residual mandatory exact misses;
5. admit optional replicas using the historical greedy current peer-byte-saving
   rule subject to rho;
6. serve effective routes, preferring an origin-local copy when present;
7. touch only physical copies that actually serve;
8. update lifecycle/accounting.

All executing effective experts are protected from same-event eviction.

Required historical parity before the sweep:

```
cache=.30, eviction=LRU, substitution=OFF
rho={0,.125,.25,.5,.75}
```

must reproduce the existing B8 decode H2D/peer/fetch coordinates exactly.

Hard stop on parity failure.

---

## S2 — full B8 CPU grid

Run exactly:

```
4 cache ratios
x 3 eviction policies
x 2 substitution modes
x 5 rho values
= 120 B8 CPU cells
```

No GPU/model timing.

For every cell report decode and full-trace:

- total/first/reload/replica fetches;
- H2D bytes;
- peer activation bytes;
- remote token-rank pairs;
- local-service fraction;
- mean/peak duplicate slots;
- mean unique resident experts;
- per-rank eviction counts;
- replica admissions;
- replica reuse before eviction;
- replica lifetime p50/p90 in layer events;
- fraction surviving >=48 layer events.

Substitution ON additionally reports:
- substituted expert/source count;
- substituted route fraction;
- substituted gate-mass fraction;
- mean/p10 similarity of accepted substitutions;
- anchor tier breakdown;
- protected-exact-miss count;
- residual exact-miss count;
- H2D avoided versus matched substitution-OFF cell.

No quality inference is allowed from similarity/gate statistics.

---

## S3 — fixed-duplicate-budget cache-relief control

A fixed rho changes the absolute number of duplicate slots when cache grows, so
add one isolation control.

Use exactly **460 duplicate slots** (historical cache30/rho=.25 cap) for every
cache ratio, eviction policy and substitution mode:

```
4 cache ratios x 3 eviction x 2 substitution = 24 cells
```

This directly asks:

> If we keep approximately the same replication amount as historical K but add
> more cache capacity around it, do unique coverage, lifetime and reloads
> improve?

Report the same metrics as S2.

The especially important comparison is:

```
cache30 + 460 duplicates
vs
cache40/50/60 + the same 460 duplicates.
```

---

## S4 — B32 secondary scaling check

Repeat the same **120-cell fractional-rho grid** on the existing/fresh B32
policy-feature trace.

Do not run the fixed-460 control on B32.

Purpose:
- determine whether larger local batch increases the value of replicas once
  cache pressure is relaxed;
- determine whether substitution changes the trend at higher communication
  volume.

No B64/B128 capture is authorized.

Total CPU cells:
```
120 B8 + 24 fixed-cap B8 + 120 B32 = 264.
```

---

## S5 — frontier and mechanism analysis

For every (batch, cache, eviction, substitution) configuration, build the raw:

```
x = peer activation bytes
y = expert H2D bytes
```

rho frontier and compute the exact first nonzero-rho byte-resource crossover:

```
lambda_first =
(H2D_rho_next - H2D_rho0) /
(peer_rho0 - peer_rho_next)
```

when a valid lower-envelope transition exists.

Historical anchor:

```
B8/cache30/LRU/exact lambda_first = 420.223139
```

Again, lambda is a byte-resource price, not a time/bandwidth multiplier.

Required mechanism plots/tables:

1. lambda_first vs cache ratio, grouped by eviction/substitution;
2. H2D vs peer Pareto fronts;
3. reload count vs cache ratio;
4. replica survival >=48 events vs cache ratio;
5. unique coverage vs duplicate count;
6. substitution fraction and residual exact misses vs cache ratio;
7. fixed-460 control showing whether extra cache rescues replica lifetime.

---

## Predeclared interpretation

### CACHE_RELIEF
At fixed 460 duplicates, increasing cache above 30% reduces H2D by >=15% while
peer bytes do not increase by >10%, or increases >=48-event replica survival by
>=2x.

### EVICTION_HEADROOM
At the same batch/cache/substitution/rho, GATE or COVERAGE Pareto-dominates LRU
in H2D/peer bytes, or reduces H2D >=10% at peer bytes within +5%.

### SUBSTITUTION_SYSTEM_HEADROOM
At the same batch/cache/eviction/rho, substitution ON reduces H2D >=10% while
peer bytes increase <=5%.

This label is **system-only** and does not imply acceptable model quality.

### REPLICATION_REGIME_SHIFT
For any exact-only or substitution-on regime, lambda_first falls by >=4x from
the historical 420.223 anchor, or a nonzero-rho cell dominates the historical
cache30/LRU counterpart.

### NO_RESCUE
None of the above holds.

Multiple labels may apply.

---

## Accuracy boundary

Substitution changes model computation, so this frozen-route replay cannot
establish output quality or autoregressive route stability.

If substitution produces promising system points, stop for owner review.
A later quality stage may evaluate a tiny fixed GSM8K subset, but no quality
run is authorized in this packet.

---

## Resource/correctness rules

CPU sweep:
- CUDA_VISIBLE_DEVICES='';
- one process;
- BLAS/OMP threads=1;
- <=8 GiB address-space.

Capture:
- only GPUs 0/1/4/5;
- pause/restore only our workers there;
- never touch 2/3/6/7;
- no timing interpretation.

Validation:
- no duplicate cap violation;
- send/recv transpose parity;
- every effective expert has a resident destination;
- active protection;
- deterministic ties;
- substitution target similarity >=.65;
- protected miss if any source route weight >=.20;
- historical exact/LRU/cache30 parity.

---

## Required outputs

- grid_points.csv/json
- fixed_duplicate_control.csv/json
- substitution_summary.csv/json
- eviction_summary.csv/json
- replica_lifecycle.csv/json
- frontier_summary.csv/json
- RESULTS.md
- validation.json
- compact figures only.

Commit the implementation/trace readiness first, then the bounded CPU result,
and stop for owner review.
