# Owner-authorized MoE-Infinity baseline repairs

Status: implementation checkpoint; native build passed; physical smoke validation running.
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

CPU unit checks: seven EAM tests passed (prefix matching, no alias, empty history, finished traces, persistent pool, per-GPU candidate budget). C++ tests added for priority versus LRU, lease protection, cold reset and prefetch rejection; all 14 residency/variant C++ tests passed.
Physical acceptance still requires observed nonzero EAM calls, actual priority evictions, peak charged bytes within all GPU budgets, cold residency receipts, BF16 output sanity and clean timing.

Latest checkpoint adds batched prefix search (equivalent to independent search), native per-GPU peak charged bytes, actual eviction/rejected-prefetch counters and hard accounting assertions. The updated 14 C++ tests pass. The initial native build passed; telemetry build is staged separately under `eam-next-build`, and must be installed atomically only after the active smoke process exits. Active `infinity_smoke1` uses the prior loaded native build and single-request EAM lookup; it is preparation/smoke only, never a primary result.

Admission/eviction race repair: exclude FETCHING/EXECUTING/pending-dispatch nodes, atomically reserve eviction victims, wait out reservation in demand enqueue, restore reservations on abort, reject duplicate pending admissions. The new abort regression initially failed and was fixed; all15 native tests now pass. The safety/telemetry build passed and was installed after stopping preparation attempt1 at completed store conversion. BF16 store conversion is complete and reusable. `infinity_smoke2` will validate the fully repaired build; no MoE-Infinity primary result yet.

Physical smoke2 reached model initialization and stopped before generation: upstream `_make_expert_nbytes_map` saw scalar offload placeholders (6 bytes/expert). Repaired it to sum canonical store-index payload sizes by expert, rather than live placeholder parameter sizes. The native topology already uses aligned canonical sizes; Python prefetch accounting now agrees. Eight Python checks pass, including this regression. Smoke3 is the next physical check.

Smoke3 passed exact budget/reset setup but found cross-device shared RoPE inputs; added input colocation outside decoder topology. Smoke4 stopped at an overly specific attention class-name guard; switched to structural self_attn paths (the upstream wrapper uses Qwen3PagedAttention). Smoke5 reached expert execution, then failed because the harness used inference_mode while native async workers perform in-place writes outside that mode. Harness now uses upstream-compatible no_grad. These are preflight failures, not timing samples.
