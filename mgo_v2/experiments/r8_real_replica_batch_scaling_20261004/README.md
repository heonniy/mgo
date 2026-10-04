# R8 cache60 realizable-replica batch scaling

This packet follows the free one-replica upper bound at R8/local-B128/cache60.

Goals:
1. Replace the free temporary replica with a capacity-preserving persistent
   replica policy.
2. Re-run the stress study at local B256 (global batch 2048).
3. Separate compute-load benefit from the price paid in H2D, D2D replica
   creation, and peer activation traffic.

Frozen common setting:
- Qwen3-30B-A3B exact routes;
- R=8;
- global cache ratio 60%;
- Gate W128 eviction;
- substitution OFF;
- decode 256;
- at most two GPU copies per expert;
- at most one new replica creation per layer-event.

B128 reuses the already frozen MATH winner:
sample=97, dp=200, BR placement=172.

B256 uses every request in each completed 2048-request exact pool. Therefore
there is no sample-seed degree of freedom. Stress search only varies:
- DP/rank assignment seed 0..255;
- BR random one-copy placement seed 0..255.

The realizable replica policy uses only current-event routing and current cache
state. It never reads future routes.

A replica consumes a real cache slot. The destination slot uses the same Gate
victim semantics as the exact cache. If a last unique copy is evicted, the
future refetch is naturally counted as H2D. Replica creation is counted as one
expert-sized GPU-to-GPU D2D copy.

No physical TPOT/E2E claim is made in this CPU packet.
