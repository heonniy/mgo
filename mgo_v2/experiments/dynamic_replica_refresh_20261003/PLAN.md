# PLAN — Dynamic stale-replica refresh

Date: 2026-10-03
Status: prospective, owner-authorized CPU headroom study.

## Why this experiment

The completed 264-cell study established two facts that now need to be
reconciled.

At B8/LRU/OFF with the **same 460 duplicate-slot cap**:

| cache | decode H2D | decode peer | reloads | replica survival >=48 |
|---:|---:|---:|---:|---:|
| 30% | 219.076 GiB | 185.297 MiB | 18,157 | 0.87% |
| 40% | 187.620 GiB | 227.117 MiB | 16,895 | 4.88% |
| 50% | 84.357 GiB | 297.496 MiB | 8,101 | 38.30% |
| 60% | 39.797 GiB | 328.332 MiB | 3,917 | 85.37% |

Thus larger cache fixes replica lifetime and reload pressure, but locality gets
worse. The historical replica rule has a structural reason: once the duplicate
cap is full, it stops admitting replicas even when an inactive duplicate is
stale and could be replaced without reducing global unique coverage.

This packet asks:

> Can duplicate-to-duplicate refresh recover communication locality without
> giving back the H2D/cache-relief benefit?

It is a frozen-route system-accounting experiment. No quality or timing claim.

---

## Frozen source state

Reuse and hash-verify:
- B8 and B32 diagnostic policy-feature traces from commit d666414;
- gate-history vectors and similarity matrix from that packet;
- the validated variable-cache replay implementation;
- historical greedy replica admission semantics.

No new GPU/model capture.

Primary workload: B32.
Secondary workload: B8.

No B64/B128, R8, NCCL timing or model execution.

---

## Frozen parameter matrix

Primary B32:

```
cache={0.40,0.60}
eviction={LRU,GATE}
substitution={OFF,ON}
rho={0.125,0.25}
```

Secondary B8:

```
cache={0.60}
eviction={LRU,GATE}
substitution={OFF,ON}
rho={0.125,0.25}
```

Substitution remains frozen:
- gate protect=.20;
- similarity=.65.

Gate history remains W=128.

No parameter tuning after results.

---

## Replica refresh semantics

A refresh may only replace a **non-executing duplicate copy** on the target
rank.

Hard invariants:
- never evict the final global copy of an expert;
- never evict a currently executing/pinned copy;
- duplicate count never exceeds the original rho cap;
- refresh does not change global unique coverage at the instant of swap;
- each new duplicate copy costs one 9-MiB H2D fetch;
- actual future evictions/reloads follow the selected eviction policy;
- local copy is preferred for service;
- all traffic savings are recomputed after every accepted swap.

The mandatory exact/substitution pipeline remains unchanged. Refresh happens
after mandatory exact admissions and before route service.

### N0 — no refresh

Exact reproduction of the completed d666414 cell. Once duplicate cap is full,
no duplicate-to-duplicate replacement.

### C1 — current refresh, max 1/event

If cap is full, enumerate legal stale-duplicate victims and absent local
replica candidates. Choose the swap with the largest **exact current-event
peer-byte reduction**. Apply at most one swap if the reduction is strictly
positive.

This is online-implementable from current routing demand.

### C2 — current refresh, max 2/event

Same as C1, recomputing exact marginal traffic after the first swap. At most two
positive-gain swaps per event.

### O4 — four-next-use oracle, max 1/event

Headroom only. For each legal swap compute exact frozen-trace peer-byte delta
over the current event plus the next four observed same-layer decode
opportunities, including loss of locality from removing the victim duplicate.
Choose the best positive-net swap. Actual cache state still evolves normally.

### OR — remaining-trace oracle, max 1/event

Same as O4 but uses all remaining observed same-layer opportunities in the
eight-decode trace. This is an upper-bound diagnostic, not an online policy.

No oracle may foresee future cache evictions when scoring; it only sees frozen
future route demand. This avoids recursively solving a global placement
problem. Actual evictions determine realized counters.

---

## Execution matrix

For B32:

```
2 cache x 2 eviction x 2 substitution x 2 rho x 5 refresh policies
= 80 cells
```

For B8:

```
1 cache x 2 eviction x 2 substitution x 2 rho x 5 policies
= 40 cells
```

Total: 120 CPU cells.

First reproduce every N0 cell byte/hash-identically with d666414.

---

## Metrics

Decode-primary, full-trace secondary:

- H2D bytes and fetch counts;
- peer activation bytes;
- remote token-rank pairs;
- reload / replica fetch counts;
- refresh admissions;
- duplicate evictions caused by refresh;
- mean unique resident experts;
- local-service fraction;
- replica age at refresh;
- replica lifetime p50/p90;
- survival >=48 layer events;
- fraction of duplicates never used again before eviction/end;
- current-event peer bytes saved per refresh;
- realized future peer bytes saved after refresh;
- substitution route/gate-mass fraction for ON cells.

Also report for every matched N0->refresh comparison:

```
delta_H2D
delta_peer
delta_reload
delta_local_service
delta_unique_coverage
```

and the two-dimensional H2D-vs-peer Pareto plane.

---

## Resource-price diagnostic

For each matched cell calculate the exact break-even resource price between N0
and each refresh policy when both deltas have opposite signs:

```
lambda_refresh =
delta_H2D_bytes / (-delta_peer_bytes)
```

This is byte-resource accounting only, not latency/bandwidth.

Evaluate descriptive J_byte at lambda={32,64,96,128,256} because the completed
B32 frontiers placed first-replication crossovers roughly in the 54--108 range
for the most relevant LRU/GATE/substitution configurations.

Do not interpret those lambda values as hardware slowdowns.

---

## Predeclared labels

### STALE_REPLICA_CONFIRMED

A current-only policy (C1 or C2) reduces peer bytes >=20% versus N0 while:
- H2D increases <=10%, and
- mean unique resident experts decreases <=1%.

### ORACLE_REFRESH_HEADROOM

O4 or OR reduces peer bytes >=25% with H2D increase <=15%, or Pareto-dominates
N0.

### ONLINE_REFRESH_PLAUSIBLE

C1/C2 achieves at least 60% of OR's peer-byte reduction from N0 while its H2D
is no more than OR H2D +5%.

### CACHE_RELIEF_PRESERVED

For cache60 refresh, H2D stays <=50% of the matched historical cache30
same-(eviction,substitution,rho) coordinate while peer traffic is lower than
the cache60 N0 coordinate.

### NO_REFRESH_HEADROOM

No current-only or oracle label above passes.

Multiple labels may apply.

---

## Required analysis

1. B32 N0 vs C1/C2/O4/OR H2D-peer frontiers.
2. B8 vs B32 refresh benefit.
3. cache40 vs cache60: does extra lifetime create more stale-replica headroom?
4. LRU vs GATE.
5. substitution OFF vs ON.
6. stale-age distribution: are refreshed victims actually old/dead?
7. how much of oracle benefit is captured by current demand alone?

The main scientific question is not whether one arbitrary lambda wins. It is
whether stale duplicate replacement moves the physical-resource frontier
toward the lower-left.

---

## Resource and validation rules

CPU only:
- CUDA_VISIBLE_DEVICES='';
- one process;
- BLAS/OMP=1;
- <=8 GiB address-space.

Required checks:
- N0 exact parity with d666414;
- active/pinned safety;
- never remove final global copy;
- duplicate cap exactness;
- unique coverage unchanged at each refresh instant;
- send/recv transpose parity;
- deterministic ties;
- substitution thresholds unchanged;
- no future information in C1/C2.

No model quality run, no E2E timing, no new capture, no extra parameter sweep.

Commit implementation/protocol first, then bounded results, and stop for owner
review.
