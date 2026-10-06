# Owner-authorized MoE-Infinity baseline repairs

Final status: priority eviction, EAM, exact expert budgets and KV lifecycle
repairs pass CPU/native/GPU regression checks and both full main workloads.
B64/L512 primary timing is stable and selected. B16/L256 bounded confirmation
completes with TTFT spread5.93% (unstable), TPOT1.48%, E2E1.53%; all samples
retained and no further automatic repeats. See confirmation_B16_L256_attempt1,
primary_B64_L512_attempt3 and ../FINAL_RESULTS.md. Historical checkpoints below
are not active work or outstanding build tasks. All queues are closed.
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

Smoke6 found output accumulation allocated on current GPU0 even when inputs were on another GPU. SetInputs now allocates on the input device and recreates its ready event when the producer device changes; output completion events now track the actual accumulation GPU/stream. Native build passed; smoke7 is running. Nine Python tests pass, including cancelling obsolete EAM prefetch queues before replacing predictions and caching immutable expert homes. No MoE-Infinity headline result yet.

Smoke7 exited without a Python traceback before producing a sample. The supervisor now records worker exit codes and enables Python fault stacks. A tiny real multi-GPU BF16 dispatcher regression reproduces invalid-resource-handle after native dispatch. Source inspection found the shared transfer-event pool reused events across GPUs; a per-device pool repair and checked event recording are building. This is not yet a validated resolution. All failed smoke/debug attempts remain separate from primary timings.

Per-device transfer-event pool build passed. The tiny real BF16 regression now passes8 cases (rows1/17, input GPUs1/3/0/2, cross-GPU expert outputs), both with launch blocking and with normal asynchronous execution/4 worker threads. Max absolute deviation from PyTorch BF16 reference is0.00010109. The same test failed before this event-pool repair. Full model smoke is next; this micro-regression is not a primary timing result.

Full-model smoke8 PASS: warmup then cold target, global4/input32/output2. Warmup EAM has no completed history (0 candidates); target uses warmup history (96 calls,122125 candidate submissions),7412 priority evictions,853 lower-priority prefetch rejections. Target starts with0 resident bytes. Per-GPU peak charged bytes exactly respect[4350541824,4350541824,4350541824,4341104640]; global peak17392730112. GPU attention/KV and finite logits checks pass. This validates mechanisms at smoke scale only; B16/L256 main workload is next.

B16/L256 primary attempt1 stopped in warmup: one expert received8649 tokens, exceeding the upstream8192-row scratch limit. No primary samples exist from this attempt. Prepared a bounded8192-row chunk executor using the same leased expert and BF16 kernels; build/8193-row native regression are pending. The larger global256/input512 workload has average8192 rows/expert under top8 routing, so skew cannot safely fit the old limit either. DeepSpeed smoke runs while this native build is prepared; no simultaneous primary timing.

Chunk build and native GPU regression PASS: rows1/17/8193 across4 producer GPUs,12 checks; largest BF16 absolute difference0.00021363. Workspaces remain8192 rows and weights stay under the same residency lease across chunks. Restart the original full workload unchanged as primary attempt2.

Full B16/L256 attempt2 completed all3 primaries with valid budget/cache/EAM receipts. Median TTFT4.88395s, TPOT3.06361s, E2E198.20504s. TTFT spread6.59% fails headline stability; retain all samples and require bounded stability follow-up before headline use. Larger B64/L512 is now running through the serial queue.

KV lifecycle repair: found local `kv` retained a previous batch cache across the next `generate`. Big warmup peak allocated~28.77–29.02GiB/GPU versus first primary~32.14–32.39GiB/GPU corroborates the extra cache. Large attempt1 stopped after repeat2; all old MoE primary attempts are superseded for headline use. New harness deletes sequence/cache references after recording tokens, collects, synchronizes, and verifies cache/tensor weakrefs are dead. Both main cells must be rerun after a short lifecycle smoke. Independent DeepSpeed/llama/OURS queue continues.

KV release smoke2 PASS: warmup and target both verify cache/tensor weakrefs released, cold expert residency0. Target EAM96calls/122114candidate submissions,7322priority evictions, global expert peak17392730112bytes and all4 per-GPU caps pass. Full unchanged main cells are now rerunning. These two-token smoke timings are not headline results.
