# Legacy migration / issue audit

Reviewed legacy tree: `MoE-Infinity-EP-archer-coslot`.

## Blocking control-plane mismatches

1. **C++/Python dispatcher API mismatch**
   - modified pybind exposes `init_slot_pool / submit_plan / wait_layer_done`;
   - Python `DistributedExpertExecutor` still calls the old
     `set_expected_queue / enqueue_expert / notify_fetch_start / wait_expert` path.
   - mgo_v2 bypasses this executor and talks to the new slot API through
     `LegacySlotExecutorAdapter`.

2. **Missing GlobalCacheController**
   - C++ comments assume a Python GlobalCacheController owns the full FetchPlan,
     but the repository does not contain that implementation.
   - `GlobalExpertController` is the replacement.

3. **Old placement is not the research method**
   - `DeviceMapManager` randomly shuffles experts;
   - `dispatch_local` uses `expert_id % total_gpus`.
   - mgo_v2 admission policies replace both.

4. **Old distributed path is not EP**
   - the legacy path sends `hidden_states.cpu()` through RPC.
   - mgo_v2 uses torch.distributed NCCL all-to-all and deduplicates one hidden
     state per token/destination rank.

5. **Multiple cache authorities**
   - old ExpertPrefetcher and Archer task scheduler can mutate residency
     independently of the controller.
   - when mgo_v2 is active:
     - do not call old prefetch/cache replacement;
     - set `MOE_EP_DISABLE_ARCHER_EVICT=1`;
     - controller is the only residency authority.

6. **Old Python ExpertCache is incompatible**
   - old priority/LFU/LRU semantics do not implement W128 gate history or
     global similarity coverage.
   - mgo_v2 implements LRU, Gate W128, and Coverage W128/k1/lambda2.

7. **Cache budget definition differs**
   - legacy C++ derives sparse cache bytes from HBM ratio;
   - paper experiments define global expert-slot ratio over L*E.
   - mgo_v2 computes explicit per-rank slot capacities and passes them to
     `init_slot_pool`.

8. **Substitution was absent**
   - mgo_v2 implements expert-level .20/.65 substitution and merges multiple
     source routes that map to the same target before expert execution.

## Low-level code retained

The modified C++ dispatcher is useful for:
- fixed resident slots;
- one staging slot;
- controller-specified victim/slot;
- asynchronous H2D into direct or staging slot;
- expert GEMM and partial combine;
- cache/fetch phase counters.

It should be treated as **rank-local**: one torchrun process per GPU. The
legacy one-process/multiple-GPU scheduler has shared state such as
`hidden_states_` and `pending_`, which is not the desired R4/R8 runtime.

## Performance blocker before final timing

The current C++ worker copies an expert from the resident slot into the
MoEMLP parameter buffer on every execution via `SetTensorsFromIds`.
That adds an expert-sized D2D weight copy even on cache hits.

Do not use final latency/speedup tables until this is removed or measured and
shown negligible. Preferred fix:

- bind GEMM parameter tensor views directly to the resident slot memory; or
- maintain slot-local parameter views and change them only when slot ownership
  changes.

This optimization should be made only after exact-EP/controller correctness is
validated on the target server.

## Prefetch

Prefetch is disabled in the initial mgo_v2 implementation. It is intentionally
separated from admission/cache policy to avoid a second residency control
loop. Reintroduce prefetch later as a **controller-issued hint** that cannot
evict or change ownership by itself.

## Current-server NUMA

Do not hard-code the old server's NUMA map. `mgo_v2.numa` derives GPU PCI
bus IDs and the corresponding Linux NUMA node at runtime and pins each rank's
CPU affinity accordingly.

On an NVSwitch 8-GPU server, NCCL handles GPU peer routing; NUMA remains
important mainly for CPU-side controller work and host-to-device expert
traffic.
