# PLAN — Future rank-affinity single-copy placement oracle

Date: 2026-10-03  
Status: prospective, owner-authorized CPU-only characterization.

## Motivation

The completed rank-local reuse study at `6a8127c` found:

- B8 four-step recurrence-weighted marginal-byte share: **73.70%**;
- B16: **78.03%**;
- B32: **80.06%**;
- but capacity-aware selective replication: **NO_HEADROOM**.

The failure mechanism is structural: optional replicas consume slots, displace
unique experts and are usually evicted before their next same-layer reuse.

This study asks whether the same rank-affinity signal is useful when the action
is changed from **add a duplicate** to **choose the owner of an unavoidable
single copy**.

## Core invariant

At every point in the replay, each resident `(layer, expert)` has **at most one
physical copy globally**.

Allowed:
- global miss/reload -> fetch one exact expert copy;
- choose the destination rank for that one copy.

Forbidden:
- duplicate replicas;
- migration of an already-resident copy;
- speculative prefetch;
- substitution;
- changing global cache capacity;
- changing eviction from LRU;
- adding a load-balancing objective.

Thus a policy may change future H2D indirectly through different per-rank
evictions, but it never pays H2D for an additional duplicate copy.

---

## Inputs and scope

CPU only. Reuse the already hash-verified exact traces:

- primary: R4 / local B8 / global B32 / cache30;
- secondary: B16/global64;
- secondary: B32/global128.

Eight decode steps only. No new model capture.

Physical cache capacities remain:

```text
[461, 461, 461, 460] slots
```

with:
- expert size = 9 MiB;
- activation row = 4096 B;
- active-expert protection;
- per-copy LRU;
- exact routing only.

Do not disturb the owner's eight resident-model GPU workers.

---

## Stage P0 — exact baseline reproduction

Reproduce the existing rho=0 F semantics exactly:

On a global miss for expert `e`, choose the currently-demanding rank with the
largest route count; ties choose the smallest rank.

For B8 the decode totals must be exactly:

```text
fetches                  17,635
H2D bytes                166,424,739,840
peer activation bytes    334,970,880
remote token-rank pairs  23,179
```

Full cache/LRU state and traffic matrices must match the independent reference
for all 432 events. Hard stop on mismatch.

---

## Stage P1 — separate current-placement quality from future affinity

Run exactly these six single-copy policies.

### F — existing baseline

```text
owner = argmax current expert-route count on requesting rank
```

### O0 — current-event exact communication oracle

At each global miss, candidate owners are **only ranks with positive current
demand for the expert**.

For every candidate rank, compute the exact current-event dispatch+combine peer
bytes after placing the single copy there, with all other current decisions
held deterministic.

Choose the candidate giving the minimum current-event peer bytes.

This answers whether the existing max-demand placement is itself a weak proxy.

### OH1 / OH2 / OH4 / OHremaining — future rank-affinity oracle

Candidate ranks remain only the ranks with positive demand in the current
event. Do not place an expert on a rank that does not currently request it.

For candidate owner `r`, score:

```text
exact peer bytes for this expert's effect
over current step + next H same-layer decode steps
```

where H is 1, 2, 4, or all remaining observed decode steps.

For scoring only:
- assume this one copy remains on candidate rank `r` through the horizon;
- freeze other experts' reference destinations when computing marginal
  dispatch-row interactions;
- include current-event cost, not only future saving.

This is an offline diagnostic score, not a claim of global optimality.

The **actual replay** remains physical:
- one copy only;
- normal LRU;
- the copy may be evicted before future reuse;
- later global misses choose again according to the policy;
- no migration while globally resident.

Ties:
1. lower predicted peer bytes;
2. larger current demand on the candidate;
3. smaller rank id.

---

## Stage P2 — actual replay accounting

For every policy and batch report:

- first-copy fetches;
- reload fetches;
- total H2D bytes;
- peer activation bytes;
- dispatch bytes;
- combine bytes;
- remote token-rank pairs;
- local-service fraction;
- mean/peak unique experts per rank;
- eviction count per rank;
- owner-choice histogram;
- fraction of miss decisions where the oracle differs from F;
- fraction of oracle choices surviving until the next same-layer demand;
- realized peer-byte saving attributable to changed owner decisions.

Also report per-step trajectories for B8:

```text
step, policy, H2D bytes, peer bytes, reloads, owner changes
```

No latency model is used in the primary comparison.

---

## Stage P3 — isolate "placement" versus "future" value

For B8 compute two deltas.

### A. Current-placement value

```text
F -> O0
```

This tells us whether exact current peer-byte minimization is already better than
the existing max-demand owner rule.

### B. Future-affinity incremental value

```text
O0 -> best(OH1, OH2, OH4, OHremaining)
```

This tells us whether knowing future rank affinity adds value beyond a better
current-event placement objective.

This separation is mandatory. Do not attribute an O0 gain to future prediction.

---

## Stage P4 — compare with the historical B8 frontier

Overlay B8:

```text
F
rho=.125
K=rho=.25
rho=.5
C=rho=.75
O0
OH1/OH2/OH4/OHremaining
```

in the same raw accounting plane:

```text
x = peer activation bytes
y = expert H2D bytes
```

Historical references:

```text
F:
  peer = 334,970,880 B
  H2D  = 166,424,739,840 B

rho=.125:
  peer = 262,500,352 B
  H2D  = 196,878,532,608 B

K=rho=.25:
  peer = 194,297,856 B
  H2D  = 235,231,248,384 B
```

The old frontier uses replication; the new policies do not.

---

## Predeclared interpretation

### FUTURE_AFFINITY_HEADROOM

At least one OH policy satisfies either:

1. relative to O0:
   - >=10% additional peer-byte reduction, and
   - H2D <= 1.05 * O0 H2D;

or

2. it dominates an old nonzero-rho point in both raw axes.

### CURRENT_ONLY_HEADROOM

O0 gives:
- >=10% peer-byte reduction versus F,
- H2D <= 1.05 * F H2D,

but no OH policy meets the FUTURE_AFFINITY_HEADROOM rule.

Interpretation: placement matters, but a future predictor may be unnecessary.

### MODEST_PLACEMENT_HEADROOM

No strong rule above, but some O0/OH policy:
- reduces peer bytes >=5% versus F with H2D <=1.05*F, or
- dominates any old nonzero-rho point.

### NO_PLACEMENT_HEADROOM

No O0/OH policy meets the above.

These are CPU accounting labels only; none imply TPOT/E2E improvement.

---

## Secondary B16/B32

Run the same six policies on the existing B16/B32 traces.

Use them only for trend consistency:

- does owner divergence from F grow with batch?
- does future affinity reduce more peer traffic at larger batch?
- does H2D stay near the corresponding single-copy F?

There are no historical K/C frontiers for B16/B32; do not invent them.

---

## Validation

Before accepting results:

1. F exactly reproduces committed B8 totals and state;
2. every expert has <=1 global copy at all times;
3. every served expert is resident on its chosen owner;
4. no admission occurs without a global miss;
5. no migration occurs while resident;
6. all rank capacities respected;
7. active experts are never evicted;
8. traffic send/receive transpose checks pass;
9. deterministic replay hashes match on repeated CPU invocation;
10. no Torch/CUDA import.

Resource bound:
- one CPU process;
- OMP/BLAS threads=1;
- `CUDA_VISIBLE_DEVICES=''`;
- <=4 GiB address-space limit.

---

## Required outputs

- `placement_points.csv/json`
- `placement_step_trajectory_B8.csv`
- `owner_choice_summary.csv/json`
- `frontier_comparison.csv/json`
- `RESULTS.md`
- `validation.json`

Compact artifacts only. Raw source receipts stay outside Git with provenance
hashes.

## Stop condition

Run exactly 6 policies x 3 available batches = **18 CPU replay cells**.

Commit results and stop for owner review.

No GPU timing, R8, longer decode, cache-ratio sweep, substitution, replication,
migration or online predictor is automatically authorized.
