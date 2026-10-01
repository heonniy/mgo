# MoE-Infinity EP+DP Multi-Process Rewrite — Skeleton Plan

## Context

The current MoE-Infinity at [`Baselines-Repository/MoE-Infinity/`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity) is a **single-process multi-GPU** system: one Python process, multiple visible CUDA devices, and a C++ `ExpertDispatcher` that orchestrates per-GPU fetch/exec threads. All cache state is shared in-process. There is no NCCL.

For research on **multi-process EP + DP** with NCCL, the user needs a new repo that:
1. Distributes work as N processes × 1 GPU each, using NCCL for all IPC.
2. Provides a **Global Cache View** as a common substrate that all GPUs can query.
3. Separates two decisions — **fetch dispatch** and **placement/relocation** — behind pluggable interfaces. Default implementations preserve MoE-Infinity's "free-form LRU cache + on-demand fetch" spirit, adapted to multi-process.
4. Provides always-on instrumentation so policies can be compared empirically later.

The goal of this first cut is a working **naive baseline** that runs Qwen3-235B-A22B end-to-end across EP × DP processes with the global cache view + interfaces in place. Smart policies are explicitly out of scope; the abstractions only need to *admit* them.

Target model: `/home/work/hyewon.lee/model/Qwen3-235B-A22B` (94 MoE layers × 128 experts × top-8, hidden=4096, moe_intermediate=1536, bf16, expert weight ≈ 36 MB, total expert weight ≈ 422 GB → offload required).

## Repo & Reuse Strategy

Create a sibling repo: **`/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity-EP/`**.

Upstream MoE-Infinity is consumed as an editable install (`pip install -e ../MoE-Infinity`) and **not modified**. We reuse:

- HF model loading + offload directory machinery in [`moe_infinity/runtime/model_offload.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/runtime/model_offload.py) (the `OffloadEngine` class, esp. `init()` and `__enter__()` setup at lines 188–608).
- The Qwen3 MoE block shim at [`moe_infinity/models/qwen.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/models/qwen.py) (`Qwen3MoEBlock` with `expert_executor`, `lib`, `layer_id` injection points). We will **substitute** a new block class but keep the same injection contract.
- The C++ extension `moe_infinity._store` per-rank:
  - `ArcherPrefetchHandle` ([`core/prefetch/archer_prefetch_handle.h`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/core/prefetch/archer_prefetch_handle.h)) — PCIe fetch via `fetch_tensors()`.
  - `ArcherTaskPool` ([`core/prefetch/task_scheduler.h`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/core/prefetch/task_scheduler.h)) — per-GPU PCIe worker threads.
  - `ExpertDispatcher` ([`core/parallel/expert_dispatcher.h`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/core/parallel/expert_dispatcher.h)) — kept *only* as a local single-GPU executor for the rank's own experts. Its multi-GPU dispatch logic is bypassed.
  - The fused `topk_softmax` kernel (used inside the Qwen block).

We **discard**:
- [`moe_infinity/distributed/expert_executor.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/distributed/expert_executor.py)'s round-robin `dispatch_local` (line 32) and the RPC code path. Replaced by our `EPExpertExecutor`.

**Critical trick**: each rank sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)` *before* `import moe_infinity`. This makes `torch.cuda.device_count() == 1` inside the C++ extension, so `kNumDevices()` ([`expert_dispatcher.cpp:35`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/core/parallel/expert_dispatcher.cpp)) returns 1 and the dispatcher spawns exactly one fetch + one exec thread for the rank's lone GPU. Without this, the C++ engine will assume 8 GPUs in every rank and break.

## File Layout

```
MoE-Infinity-EP/
├── pyproject.toml                 # deps: moe_infinity (-e ../MoE-Infinity), torch, transformers, nvtx, pyyaml
├── configs/
│   ├── qwen3_235b_ep8_dp1.yaml
│   └── qwen3_235b_ep4_dp2.yaml
├── moe_infinity_ep/
│   ├── launch/
│   │   ├── distributed_setup.py   # parse torchrun env, build EP/DP NCCL groups, ProcessTopology
│   │   └── entry.py               # MoE_EP top-level (mirrors upstream entrypoints/big_modeling.py:MoE)
│   ├── runtime/
│   │   ├── distributed_engine.py  # DistributedOffloadEngine(OffloadEngine) — subclass + swap executor
│   │   └── ep_executor.py         # EPExpertExecutor.run_layer(...)
│   ├── cache/
│   │   ├── view.py                # GlobalCacheView, CacheState, ExpertSlot
│   │   └── sync.py                # sync_and_classify(): EP all-gather demand + deterministic classify
│   ├── policies/
│   │   ├── base.py                # FetchDispatchPolicy / PlacementPolicy ABCs, FetchOp / PlacementOp
│   │   ├── registry.py            # name -> class, config-driven
│   │   ├── naive_fetch.py         # NaiveFetchPolicy (target_rank = expert_id % ep_size)
│   │   └── naive_placement.py     # NaiveTokenRoutePlacement (hit: route token; miss: place on target + LRU evict)
│   ├── exec/
│   │   ├── layer_loop.py          # per-layer choreography (pseudocode in §4)
│   │   ├── nvlink_router.py       # token activation all-to-all on EP group (padded equal-size first)
│   │   └── pcie_fetcher.py        # wraps archer_engine.fetch_tensors as an awaitable
│   ├── instrument/
│   │   ├── counters.py            # Counters dataclass + timers (always on)
│   │   └── trace.py               # gather_object to rank 0, dump JSON + CSV
│   ├── models/
│   │   └── qwen_ep.py             # Qwen3MoEBlockEP — replaces upstream block via class swap in __enter__
│   └── utils/config.py            # YAML loader -> Config dataclass
├── scripts/
│   ├── run_ep8_dp1.sh             # torchrun --standalone --nproc_per_node=8 ...
│   └── verify_vs_upstream.py
└── tests/
    ├── test_global_cache_view.py
    ├── test_policies.py
    └── test_distributed_setup.py
```

## Key Abstractions

### Process topology (`launch/distributed_setup.py`)

```python
@dataclass(frozen=True)
class ProcessTopology:
    world_size: int          # auto-detected = torch.cuda.device_count() before VISIBLE_DEVICES gate
    ep_size: int             # from config
    dp_size: int             # from config; ep_size * dp_size must == world_size
    global_rank: int
    ep_rank: int             # = global_rank % ep_size
    dp_rank: int             # = global_rank // ep_size
    local_device: int        # always 0 after CUDA_VISIBLE_DEVICES gate
    ep_group: dist.ProcessGroup    # ranks with same dp_rank
    dp_group: dist.ProcessGroup    # ranks with same ep_rank
```

Layout convention: **EP varies fastest**. Default config sets `ep_size = world_size, dp_size = 1`; user overrides for EP+DP runs.

### Global Cache View (`cache/view.py`)

```python
@dataclass
class ExpertSlot:
    layer_id: int
    expert_id: int
    last_used_seq: int       # global lockstep counter (see below)

@dataclass
class CacheState:
    capacity_per_rank: int
    cache: List[OrderedDict[Tuple[int,int], ExpertSlot]]   # per ep_rank, LRU = first
    locate_index: Dict[Tuple[int,int], Set[int]]           # (layer, expert) -> {ranks}
    layer_seq_counter: int   # increments per sync_and_classify call, identical across EP group

@dataclass
class ClassifyResult:
    hits: Dict[Tuple[int,int], int]      # (layer, expert) -> serving rank (lowest if multi-cached)
    misses: List[Tuple[int,int]]         # deterministic order (sorted by expert_id)
    demand_vector: torch.Tensor          # union, [num_experts] int8
    per_rank_demand: torch.Tensor        # [ep_size, num_experts] from all-gather

class GlobalCacheView:
    def sync_and_classify(self, layer_id, local_router_mask) -> ClassifyResult: ...
    def locate(self, layer_id, expert_id) -> Optional[int]: ...
    def free_slots(self, ep_rank) -> int: ...
    def apply_placements(self, ops: List[PlacementOp]) -> None:
        """Deterministic mutation; produces byte-identical state on every EP rank."""
    def touch(self, ep_rank, layer_id, expert_id) -> None: ...
```

**Sync protocol** (called once per MoE layer per token-position):
1. `local_demand = (router_mask.sum(dim=0) > 0).to(torch.int8)` → `[num_experts]`.
2. `dist.all_gather` into `per_rank_demand` on **`ep_group`** (not world; see §3 on EP+DP).
3. `union_demand = per_rank_demand.any(dim=0)` — byte-identical on every rank.
4. Walk experts in `expert_id` order, classify hit/miss using `locate_index`. Multi-cached → lowest rank wins (deterministic).
5. Increment `layer_seq_counter` once at end (identical across ranks).

**Deterministic LRU trick** (avoids any further sync): on every touch (hit or install), set `last_used_seq = layer_seq_counter * 1_000_000 + expert_id`. Eviction picks the entry with smallest `last_used_seq`. Because `layer_seq_counter` advances in lockstep and demand vectors are byte-identical, every rank computes the same eviction victim for any candidate rank's cache. **No additional NCCL sync for placement/eviction is needed.**

### Policy interfaces (`policies/base.py`)

```python
@dataclass
class FetchOp:
    layer_id: int
    expert_id: int
    target_rank: int          # which rank will host this expert post-fetch
    lane: int = 0             # PCIe lane / stream hint, advisory

@dataclass
class PlacementOp:
    layer_id: int
    expert_id: int
    target_rank: int
    source: Literal["pcie", "nvlink"]    # naive default uses "pcie" only
    source_rank: Optional[int] = None
    evict_key: Optional[Tuple[int,int]] = None

@dataclass
class LaneLoads:
    pcie_bytes_outstanding: List[int]    # per ep_rank
    nvlink_bytes_outstanding: List[int]

class FetchDispatchPolicy(ABC):
    @abstractmethod
    def decide(self, misses, lane_loads, cache_view) -> List[FetchOp]: ...

class PlacementPolicy(ABC):
    @abstractmethod
    def decide(self, classify_result, fetch_ops, cache_view) -> List[PlacementOp]: ...
```

Both are **pure functions**: input → plan. The executor does I/O.

**Naive defaults**:
- `NaiveFetchPolicy`: `target_rank = expert_id % ep_size` (mirrors upstream `expert_executor.py:57`). Lane = 0.
- `NaiveTokenRoutePlacement`:
  - For **hits**: emit NO `PlacementOp` (expert stays put; token will be routed via NVLink in step 5 of layer loop).
  - For **misses**: `PlacementOp(target_rank = expert_id % ep_size, source="pcie")`. If `free_slots(target_rank) == 0`, set `evict_key` to the rank's LRU entry.

## Per-Layer Choreography (`exec/layer_loop.py`)

```python
def run_moe_layer(layer_id, hidden_states, gate, ctx):
    # 1. Router locally on each rank's tokens (EP rank input is replicated; DP rank has its shard).
    router_logits = gate(hidden_states)
    router_mask, routing_weights = ctx.lib.topk_softmax(router_logits)

    # 2. Cache view sync — EP all-gather demand vector (~128 ints)
    with ctx.counters.time("cache_sync_us"):
        classify = ctx.cache_view.sync_and_classify(layer_id, router_mask)

    # 3. Policies plan (pure, deterministic across EP ranks)
    with ctx.counters.time("policy_us"):
        fetch_ops = ctx.fetch_policy.decide(classify.misses, ctx.lane_loads, ctx.cache_view)
        placement_ops = ctx.placement_policy.decide(classify, fetch_ops, ctx.cache_view)

    # 4a. Start PCIe fetches for this rank only (those targeting my_ep_rank)
    my_fetches = [op for op in fetch_ops if op.target_rank == ctx.topology.ep_rank]
    with ctx.streams.pcie:
        ctx.pcie_fetcher.start(my_fetches)         # archer_engine.fetch_tensors per expert

    # 4b. NVLink token routing for hits (and for misses, once fetched)
    #     Build send buffer: tokens packed by destination ep_rank.
    expert_to_rank = {**classify.hits,
                      **{(op.layer_id, op.expert_id): op.target_rank for op in fetch_ops}}
    a2a_send, a2a_meta = pack_tokens(hidden_states, router_mask, routing_weights, expert_to_rank)
    with ctx.streams.comm:
        a2a_recv = ctx.nvlink_router.all_to_all(a2a_send, a2a_meta, group=ctx.topology.ep_group)
        ctx.counters.bump("nvlink_activation_bytes", a2a_send.numel() * a2a_send.element_size())

    # 5. Synchronize: fetches must complete before local exec consumes them.
    ctx.pcie_fetcher.wait()
    ctx.cache_view.apply_placements(placement_ops)   # deterministic; no NCCL

    # 6. Local expert compute on received tokens — delegate to upstream C++ ExpertDispatcher
    #    (running in single-GPU mode for this rank)
    local_outputs = ctx.local_exec.run(received_tokens=a2a_recv,
                                       received_weights=a2a_meta.weights,
                                       local_expert_ids=a2a_meta.local_expert_ids)

    # 7. All-to-all back (return outputs to originating ranks)
    with ctx.streams.comm:
        out_recv = ctx.nvlink_router.all_to_all_back(local_outputs, a2a_meta,
                                                      group=ctx.topology.ep_group)

    # 8. Combine: weighted sum of expert outputs per token
    final = unpack_and_combine(out_recv, a2a_meta, routing_weights)
    ctx.counters.bump("tokens_processed", hidden_states.shape[0])
    return final, router_logits
```

**Overlap (first cut: best-effort only, correctness first)**: PCIe fetch (4a) runs on its own CUDA stream and the C++ task-pool thread; token-routing all-to-all (4b) runs on the NCCL stream. Both can be in flight together. Local exec waits on `pcie_fetcher.wait()`. Full pipeline overlap (layer N+1 sync behind layer N exec) is a Milestone 7 item.

## EP + DP Semantics

- World size = `torch.cuda.device_count()` auto-detected; user config splits into `ep_size × dp_size`. Default in YAML: `ep_size = world, dp_size = 1`.
- **EP group**: ranks with the same `dp_rank`. The all-gather of demand vectors, the token all-to-all, and the cache view replication all live on `ep_group`. Two DP slabs do NOT share cache view state.
- **DP group**: ranks with the same `ep_rank`. Independent EP slabs each see a different batch shard. Each DP slab loads its own expert weights into its cache (no cross-slab sharing in the naive version).
- **Within one EP group**: all ranks see the **same input tokens** (replicated). The router computes redundantly on every EP rank — cheap. Each rank only executes the experts that the placement says it owns; the all-to-all combines outputs back to all ranks identically. This matches upstream MoE-Infinity's single-process behavior most closely and gives bitwise-identical numerics for verification.
- **Across DP groups**: each DP slab processes a different batch shard. At generation entry, `MoE_EP.generate` shards the input batch DP-ways; at exit, gathers outputs from rank 0 of each DP slab.

For Milestone 1's recommended config (8 visible GPUs), default to `ep_size=8, dp_size=1`. EP+DP path (`ep_size=4, dp_size=2`) is configured by swapping the YAML — the code paths are the same.

## Model Integration

Subclass upstream `OffloadEngine` rather than monkey-patching.

```python
# runtime/distributed_engine.py
class DistributedOffloadEngine(OffloadEngine):
    def __init__(self, capacity, archer_cfg, topology, ep_cfg):
        super().__init__(capacity, archer_cfg)
        self.topology = topology
        self.ep_cfg = ep_cfg

    def init(self, cls, ar_config):
        super().init(cls, ar_config)
        # Swap executor (super() set self.expert_executor at line 188)
        self.expert_executor = EPExpertExecutor(
            archer_engine=self.archer_engine,
            local_dispatcher=self.expert_dispatcher,        # reused per-rank, single-GPU
            cache_view=GlobalCacheView(self.topology, self.ep_cfg.cache_capacity, ...),
            fetch_policy=policy_registry.load(ep_cfg.policies.fetch_dispatch),
            placement_policy=policy_registry.load(ep_cfg.policies.placement),
            topology=self.topology,
            counters=Counters(),
        )
        return self

    def __enter__(self):
        # Override block class binding before super().__enter__() runs HF model construction
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qmoe
        qmoe.Qwen3MoeSparseMoeBlock = Qwen3MoEBlockEP    # our shim, NOT upstream's
        return super().__enter__()
```

`Qwen3MoEBlockEP` ([`models/qwen_ep.py`](MoE-Infinity-EP/moe_infinity_ep/models/qwen_ep.py)) is the same as upstream `Qwen3MoEBlock` except its `forward()` calls `self.expert_executor.run_layer(layer_id, hidden_states, gate=self.gate)` and returns the result. The new `run_layer` API replaces upstream's `dispatch_local` + `wait_dispatch_local` pair.

`MoE_EP` entry ([`launch/entry.py`](MoE-Infinity-EP/moe_infinity_ep/launch/entry.py)) mirrors upstream `MoE` ([`entrypoints/big_modeling.py:24`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/entrypoints/big_modeling.py)) but uses `DistributedOffloadEngine` and gates first-time offload to global rank 0 with a file lock around the offload directory (other ranks barrier-wait, then read the cached `name_id_map.json`).

## Instrumentation (`instrument/counters.py`)

Always on. Per-rank instance held by `EPExpertExecutor`.

```python
@dataclass
class Counters:
    pcie_fetch_bytes: int = 0
    nvlink_expert_migration_bytes: int = 0
    nvlink_activation_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    tokens_processed: int = 0
    layer_compute_us: List[int] = field(default_factory=list)
    cache_sync_us: List[int] = field(default_factory=list)
    a2a_us: List[int] = field(default_factory=list)
    pcie_wait_us: List[int] = field(default_factory=list)
    expert_access_trace: List[Tuple[int, int, int, int]] = field(default_factory=list)
        # (step, layer, ep_rank, expert_id)

    def bump(self, name, val=1): ...
    def time(self, name) -> ContextManager: ...
```

At end of `MoE_EP.generate`: `dist.gather_object` per-rank Counters dicts to global rank 0, which writes a single JSON + CSV per run to `cfg.instrumentation.trace_path`. NVTX ranges wrap each phase for Nsight profiling.

`pcie_fetch_bytes` is incremented inside `pcie_fetcher.start()` by summing the byte size of each fetched expert (queryable from the tensor index in `archer_engine`).

## Milestones

| # | Goal | Acceptance criterion |
|---|------|---------------------|
| 1 | Distributed scaffolding + weight load | 8 procs come up, all load Qwen3-235B, barrier, exit clean |
| 2 | Single-rank loopback (`ep_size=1`) | End-to-end forward through naive executor matches upstream output bitwise (no NCCL exercised) |
| 3 | Full naive EP=world_size, DP=1 | End-to-end generation matches upstream output token-for-token (greedy) for 32-token prompt × 8 tokens out |
| 4 | Instrumentation | JSON trace dumped; totals sanity-check (pcie_bytes ≈ #misses × 36MB) |
| 5 | Policy registry + ABC compliance | Naive policies load through registry; a no-op smoke policy plugs in cleanly |
| 6 | DP > 1 (EP=4, DP=2 on 8 GPUs) | Per-DP-slab outputs each match the EP=8 DP=1 reference for the corresponding batch shard |
| 7 | Overlap & perf tuning | Pipeline cache sync of layer N+1 behind layer N compute; tune NCCL parameters |

Milestone 3 = the "naive baseline complete" point and is the primary deliverable of this plan. Everything after is hardening / extensibility.

## Verification (`scripts/verify_vs_upstream.py`)

The naive default should produce **bitwise-identical outputs** to upstream single-process MoE-Infinity for greedy decoding, because:
- Router runs locally on the same hidden states (EP-replicated input).
- Expert MLP forward is the same kernel.
- All-to-all is a lossless permutation (no reductions).
- No floating-point reduction reorderings are introduced.

Procedure:
1. Pick a fixed 32-token prompt. Run upstream single-process (8 visible GPUs) with `do_sample=False, max_new_tokens=8` → record `decoded_upstream` and `logits_upstream[0]`.
2. Run EP=8 DP=1 multi-process on the same 8 GPUs with the same seed → record `decoded_ep` and gather `logits_ep[0]` to rank 0.
3. Assert `decoded_ep == decoded_upstream` token-for-token.
4. Assert `logits_ep[0]` is bitwise equal (or within 1 ULP for bf16) to `logits_upstream[0]`.
5. For DP > 1: run `ep_size=4, dp_size=2` with two prompts; each DP slab's output must match the reference run on its prompt.

Debug-only assertion in `apply_placements`: hash the cache state on each EP rank and `dist.all_reduce(MAX-MIN)` to verify byte-identity. Run periodically in dev mode.

## Open Risks

1. **`kNumDevices` timing** — must set `CUDA_VISIBLE_DEVICES=$LOCAL_RANK` before `import moe_infinity` or the per-rank C++ engine misbehaves.
2. **First-time offload thrash** — 422 GB to disk on first run. Gate writes to global rank 0 with a file lock; other ranks barrier-wait, then read cached `name_id_map.json` (upstream supports this idempotently at [`model_offload.py:439`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/runtime/model_offload.py)).
3. **Miss latency at Qwen3-235B scale** — naive may take seconds per token. Expected; the policy framework is what fixes it. Keep verification prompts tiny (32 tokens × 2 new tokens).
4. **All-to-all token-bucket sizes differ across ranks** — start with **padded equal-size all-to-all** (pad to per-pair max, send `valid_count`). Switch to uneven `all_to_all_single` only in Milestone 7.
5. **Determinism drift** — single biggest correctness risk. Sort all dict iterations by integer keys before NCCL ops; cast bools/ints to fixed dtype; debug assertion above.
6. **Generation loop sync overhead** — 94 layers × N decode tokens × per-layer all-gather = many NCCL launches. Each is small (~50 µs) but adds up. Acceptable for correctness milestones; batch-fuse demand sync across all layers in prefill at Milestone 7.
7. **C++ dispatcher LRU vs GlobalCacheView LRU may diverge** — the per-rank C++ engine has its own LRU. As long as we only enqueue experts the policy has classified as resident, the C++ cache-hit check matches. Add a debug assertion that compares the two after each layer.

## Critical Files

To be created (under `MoE-Infinity-EP/`):
- `moe_infinity_ep/launch/distributed_setup.py` — `ProcessTopology`, EP/DP group construction
- `moe_infinity_ep/launch/entry.py` — `MoE_EP` top-level
- `moe_infinity_ep/runtime/distributed_engine.py` — `DistributedOffloadEngine(OffloadEngine)`
- `moe_infinity_ep/runtime/ep_executor.py` — `EPExpertExecutor.run_layer`
- `moe_infinity_ep/cache/view.py` + `sync.py` — `GlobalCacheView` + sync protocol
- `moe_infinity_ep/policies/base.py` + `registry.py` + `naive_fetch.py` + `naive_placement.py`
- `moe_infinity_ep/exec/layer_loop.py` + `nvlink_router.py` + `pcie_fetcher.py`
- `moe_infinity_ep/models/qwen_ep.py` — `Qwen3MoEBlockEP`
- `moe_infinity_ep/instrument/counters.py` + `trace.py`
- `scripts/run_ep8_dp1.sh` + `verify_vs_upstream.py`

To be referenced (read-only, in upstream):
- [`moe_infinity/runtime/model_offload.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/runtime/model_offload.py) — `OffloadEngine` lines 173–608 (executor + dispatcher setup), 806–920 (forward hooks), 906–911 (register_expert)
- [`moe_infinity/distributed/expert_executor.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/distributed/expert_executor.py) lines 32–66 (the dispatch pattern we replace)
- [`moe_infinity/models/qwen.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/models/qwen.py) — the block class to subclass
- [`core/parallel/expert_dispatcher.h`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/core/parallel/expert_dispatcher.h) lines 30–100 — public API we still call (set_inputs, enqueue_expert, notify_fetch_start, wait_expert)
- [`moe_infinity/entrypoints/big_modeling.py`](/home/work/hyewon.lee/Baselines-Repository/MoE-Infinity/moe_infinity/entrypoints/big_modeling.py) — `MoE` class as the entry point template
