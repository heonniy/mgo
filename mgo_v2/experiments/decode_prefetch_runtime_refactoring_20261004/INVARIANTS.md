# Correctness invariants

These invariants are mandatory. Performance results are invalid if any one
fails.

## A. Global residency

**A1 — single-copy main residency**

For every expert key `(layer, expert)`, at most one MAIN owner exists
globally. Replication is out of scope.

**A2 — at most one prefetch reservation**

A key absent from MAIN may appear in at most one rank's PREFETCH slot.

**A3 — no MAIN/PREFETCH duplication**

A key may not be simultaneously MAIN and PREFETCH. Candidate selection must
exclude the post-current-plan MAIN resident set.

## B. Capacity

For every rank at every event:

```
#MAIN roles == C
#PREFETCH roles == P
#physical expert slots == C + P
```

Role swaps preserve both counts exactly.

## C. Replicated controller

Every rank starts an event with byte-identical logical:
- main cache shadow;
- prefetch reservation shadow;
- slot roles;
- predictor table/config;
- BR/CA/LA policy state.

Given the same metadata packet, every rank derives the same:
- current hit/miss partition;
- current miss owner assignment;
- victims;
- prefetch candidates;
- prefetch owners;
- promotions/discards.

Production mode has no plan broadcast. Debug mode may all-gather a tiny plan
hash and must assert equality.

Physical H2D readiness is rank-local and is **not** part of the deterministic
logical shadow.

## D. Prefetch state machine

Allowed states for a PREFETCH-role slot:

```
EMPTY
QUEUED(key, target_layer)
INFLIGHT(key, target_layer)
READY(key, target_layer)
```

At the target layer:
- demanded READY -> promote;
- demanded QUEUED -> promote + raise queue priority;
- demanded INFLIGHT -> promote + wait on the existing transfer event;
- unused QUEUED -> cancel before staging if possible;
- unused INFLIGHT -> let complete safely, then discard;
- unused READY -> discard.

A demanded prefetched key must never trigger a duplicate CPU->GPU copy.

## E. Promotion

Promotion occurs on the **same rank that owns the prefetch slot**.

If a MAIN-role slot is empty:
- prefetched slot changes PREFETCH -> MAIN;
- empty main-role slot changes MAIN -> PREFETCH_EMPTY.

If MAIN is full:
- choose a legal main-cache victim using the active eviction policy;
- evict the victim from the logical main cache;
- prefetched slot changes PREFETCH -> MAIN;
- victim slot changes MAIN -> PREFETCH_EMPTY.

No 9-MiB device copy is performed for promotion.

The promoted rank becomes the execution owner for that layer. This is why
prefetch rank placement is evaluated under BR/CA/LA.

## F. Current-demand priority

Current-layer demand always outranks speculative work.

```
demand miss / demanded pending-prefetch > unused next-layer prefetch
```

A wrong prefetch must never delay a current demand transfer when an urgent
transfer is queued.

## G. Prefill boundary

New predictor/prefetch logic is disabled in prefill.

Prefill uses the existing main cache and mandatory miss admission. For all
BR/CA/LA policies, per-layer mandatory miss fetch counts must satisfy:

```
max(rank_fetches) - min(rank_fetches) <= 1
```

No decode prefetch slot may contain a valid key at the end of prefill.

## H. Native routing

Prediction affects only data movement and residency. It never changes the
model router output, selected experts, routing weights, or generated token.

A wrong prediction may cost bandwidth/capacity, but must not change numerical
semantics.

## I. Transport

Optimized expert execution has exactly:
- one metadata collective (small, separate);
- one fused forward A2A;
- one return A2A.

No hidden/weight/expert-id secondary A2A is allowed in the final fast path.
No return-metadata A2A is allowed.

## J. Baseline parity

With:
- prefetch P=0;
- new communicator disabled;
- optimized H2D scheduler disabled;

the refactored controller/cache path must reproduce the frozen baseline:
- token hash;
- hit/miss counts;
- owners;
- victims;
- H2D bytes;
- cache trajectory.

Every milestone has to keep an explicit parity test.
