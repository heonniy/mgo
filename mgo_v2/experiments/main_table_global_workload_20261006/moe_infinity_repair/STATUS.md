# Owner-authorized MoE-Infinity baseline repairs

Status: implementation checkpoint; native build and physical validation pending.
Upstream MoE-Infinity: 9f819a6d43e043bded6e0692e5e58793e1623364.
moe-store: v0.2.2, 096f51f92ca9d698907d9a79120a7819ea094266.

Owner explicitly requests functioning priority eviction, EAM and expert budgets.
Report this as repaired MoE-Infinity, never as unmodified upstream.

- Qwen3 wrapper invokes per-request EAM after actual route submission.
- EAM matches observed layer prefix against completed traces; no future routing input. Empty history gives no speculative candidates. Completed requests populate history and release live entries.
- Lowest predicted priority is evicted, with LRU/id tie breaking. Demand/transfer/execution leases remain protected. A lower/equal-priority prefetch cannot displace higher/equal-priority residency.
- Exact per-GPU expert byte limits use the shared native residency authority, with CACHE admission in both phases (no transient over-budget slot). Prefetch candidate lists are bounded by those limits.
- New quiescent cache clear refuses live tickets/leases, clears physical residency while retaining capacity. The old reset_cache only cancels queues.
- Preserve warmup EAM history and restore its snapshot before every measured repeat, preventing earlier target repeats from training later ones. Clear expert residency before each measured batch, retain CPU source and compiled code.

CPU unit checks: five EAM tests passed (prefix matching, no alias, empty history, finished traces, persistent pool, per-GPU candidate budget). C++ tests added for priority versus LRU, lease protection, cold reset and prefetch rejection; not run yet.
Physical acceptance still requires observed nonzero EAM calls, actual priority evictions, peak charged bytes within all GPU budgets, cold residency receipts, BF16 output sanity and clean timing.
