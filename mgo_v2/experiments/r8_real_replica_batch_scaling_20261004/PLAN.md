# PLAN — realizable hot-expert replication at B128 and B256

## 1. Questions

A. How much of the prior B128 free-replica compute headroom (BR 12.87%, CA
12.39%) survives once cache capacity, victims, persistence and creation cost
are modeled?

B. Does increasing local batch from 128 to 256 increase the value of hot-expert
replication?

C. Does communication-aware one-copy placement remain weaker than replica-aware
compute balancing in the high-cache/high-batch regime?

## 2. Realizable online replica policy

After mandatory exact misses are admitted for the current layer-event:

1. Read only current routed demand.
2. Compute the current per-rank expert-row load under the resident copies.
3. Consider active experts with exactly one resident copy.
4. A hot expert is eligible when total current rows >= hotness_multiplier *
   local_batch.
5. For every candidate destination rank without that expert:
   - find a real evictable slot using the same Gate W128 victim rule;
   - estimate the current-event max-rank-row reduction if the expert obtains a
     second execution site;
   - require a minimum fractional reduction in current max-rank load.
6. Create at most one best replica in the event.
7. The replica persists in the actual cache until later eviction.
8. Route current token-expert rows greedily to the least-loaded resident copy,
   using local origin only as a tie-breaker.

The policy never uses future routing.

## 3. Real costs/counters

A created replica:
- occupies one real cache slot;
- can evict a unique copy or an older replica;
- copies EXPERT_BYTES from an existing owner to the new owner, counted as D2D;
- can increase future H2D if its victim is later reloaded;
- can be reused in later events;
- changes actual peer activation traffic according to chosen execution owner.

Required output:
- decode sum/peak max-rank expert rows;
- mean rank-load CV;
- exact global hit;
- H2D bytes and delta from one-copy;
- D2D replica-creation bytes;
- peer bytes;
- replica creations;
- replica-served rows;
- unique-copy victims vs duplicate-copy victims;
- dynamic duplicate-copy occupancy.

## 4. Policy grid

To avoid choosing one arbitrary threshold after seeing results, predeclare:

hotness multiplier:
- 1.0x local batch;
- 1.5x;
- 2.0x;
- 3.0x.

minimum current-event max-load reduction:
- 2%;
- 5%.

global duplicate-copy budget as fraction of total cache slots:
- 1%;
- 2%;
- 5%.

This is 24 online policies per base placement policy and batch.

At most two copies of an expert are allowed.

Report three views instead of collapsing resources into one weighted objective:
1. maximum compute-balance improvement;
2. best improvement with H2D increase <=2%;
3. Pareto frontier of critical rows vs H2D, with D2D/peer traffic annotated.

## 5. B128

Reuse the frozen stress winner from
r8_b128_c60_replication_stress_20261004:
- dataset MATH;
- sample_seed=97;
- dp_seed=200;
- placement_seed=172.

Evaluate BR and deterministic CA one-copy baselines plus the full realizable
policy grid.

## 6. B256 stress selection

R8*B256=2048, exactly the candidate-pool size. Therefore all requests are used.

For MATH and ShareGPT:
- placement_seed 0..255;
- dp_seed 0..255.

Stage A:
- use 48 proxy events;
- retain the 8 worst placement seeds by BR critical-rank-row proxy;
- for each, retain the 2 DP seeds with largest peer proxy.

Stage B:
- exact Gate replay the 16 candidates per dataset;
- select one winner using BR one-copy only:
  1. max decode sum(max-rank expert rows);
  2. max decode peer bytes;
  3. max peak max-rank rows;
  4. deterministic dataset/seed tie order.

After the winner is frozen, evaluate CA and the same realizable replica grid.

Also compute the free one-replica oracle on the B256 winner for direct comparison
with B128 upper-bound headroom.

## 7. Interpretation

A positive result requires more than lower rank rows. The useful regime is where
persistent replication preserves a meaningful fraction of the oracle compute
headroom without paying it back through large H2D growth.

Because H2D is the dominant offload cost, highlight variants with <=2% H2D
increase separately. Do not call a policy "faster" until physical timing exists.

## 8. Stop

This packet is CPU-only. Commit results and stop.
Do not automatically launch a 4-GPU microbenchmark or physical inference.
