"""DistributedOffloadEngine — subclass of upstream OffloadEngine.

Reuses upstream's archer_engine setup, expert_dispatcher construction, and
model-construction patches.  Replaces the per-process expert_executor with
our EP-aware variant and wires the GlobalCacheController as the single
source of truth for cache decisions.

Architecture invariants (DO NOT regress):
  * For ep_size > 1 the controller MUST be wired before the first forward.
    No legacy hit/miss path; no pin_manager.
  * archer's autonomous LFU evict is disabled at process start
    (MOE_EP_DISABLE_ARCHER_EVICT=1, set in launch/entry.py).  The controller
    issues explicit_evict for every eviction.
  * archer's C++ trace-based prefetch is OFF by default.  Python issues
    archer prefetch via CommandDispatcher only — no second control loop.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Type, Union

# NOTE: this module must be imported AFTER launch.distributed_setup.pin_visible_device()
# has run, because moe_infinity's C++ extension probes torch.cuda.device_count()
# at import time.  See launch/distributed_setup.py.
from moe_infinity.runtime import OffloadEngine
from moe_infinity.utils import ArcherConfig

from .ep_executor import EPExpertExecutor
from ..utils.expert_specs import resolve_expert_bytes


class DistributedOffloadEngine(OffloadEngine):
    """OffloadEngine specialised for one process per GPU + NCCL EP/DP.

    Construction is identical to upstream's two-phase pattern:
        engine = DistributedOffloadEngine(capacity, model_config, topology, ep_cfg)
        with engine.init(cls=model_cls, ar_config=cfg):
            model = model_cls.from_pretrained(...)
    """

    def __init__(self, capacity, config, topology, ep_cfg: Dict[str, Any] | None = None):
        super().__init__(capacity, config)
        self.topology = topology
        self.ep_cfg = ep_cfg or {}

    def init(self, cls: Type, ar_config: Union[str, Dict, ArcherConfig]):
        from ..instrument.counters import Counters

        ep_size = self.topology.ep_size
        num_experts = int(getattr(self.config, "num_experts", 0))
        num_layers = int(getattr(self.config, "num_hidden_layers", 0))

        # CRITICAL: archer reads its capacity knobs from environment variables
        # in static helpers (_host_memory_ratio, _sparse_cache_ratio,
        # MOE_INFINITY_SPARSE_BYTES) at HostMemoryPool / ArcherPrefetchHandle
        # construction time inside super().init().  Propagate the Python-side
        # config to those env vars BEFORE super().init so the two sides agree.
        # Skipping this leaves the YAML fields silently inert (memory says
        # host_memory_ratio was inert pre-2026-05-26; this fix wires it).
        capacity_per_rank = None
        # coslot: build the controller for ep_size>=1.  The single-rank path now
        # also runs through the controller + archer submit_plan (no legacy
        # enqueue API), so ep_size==1 needs a cap + slot pool too.
        if ep_size >= 1 and num_experts > 0 and num_layers > 0:
            capacity_per_rank = self._resolve_capacity(num_layers, num_experts)
            self._sync_archer_capacity_env(capacity_per_rank, num_layers, num_experts)
        self._sync_archer_host_ratio_env(ar_config)

        # Upstream init() builds self.archer_engine and assigns
        # self.expert_executor = DistributedExpertExecutor(...) at line 188.
        super().init(cls, ar_config)

        counters = Counters()
        controller = None
        if capacity_per_rank is not None:
            controller = self._build_controller(
                ep_size, num_layers, num_experts, capacity_per_rank,
            )

        self.expert_executor = EPExpertExecutor(
            archer_engine=self.archer_engine,
            local_dispatcher=None,  # set later via set_expert_dispatcher
            topology=self.topology,
            counters=counters,
            controller=controller,
        )
        return self

    # ---- archer env propagation (host_memory_ratio + sparse cap) ----

    def _sync_archer_host_ratio_env(self, ar_config) -> None:
        """Propagate ``host_memory_ratio`` to ``MOE_INFINITY_HOST_MEMORY_RATIO``.

        archer's HostMemoryPool reads ONLY the env var; the YAML/ArcherConfig
        field is silently inert without this propagation.  Accepts ``ar_config``
        as either a dict (the typical path through entry.py) or as a parsed
        ArcherConfig dataclass.
        """
        ratio = 0.0
        try:
            if isinstance(ar_config, dict):
                ratio = float(ar_config.get("host_memory_ratio", 0.0) or 0.0)
            else:
                ratio = float(getattr(ar_config, "host_memory_ratio", 0.0) or 0.0)
        except Exception:
            return
        if ratio <= 0.0 or ratio > 1.0:
            return
        # Don't overwrite an explicit env override from the user.
        if not os.environ.get("MOE_INFINITY_HOST_MEMORY_RATIO"):
            os.environ["MOE_INFINITY_HOST_MEMORY_RATIO"] = f"{ratio:.4f}"
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[distributed_engine] MOE_INFINITY_HOST_MEMORY_RATIO "
                      f"= {ratio:.4f} (propagated from YAML)", flush=True)

    def _sync_archer_capacity_env(
        self, capacity_per_rank: int, num_layers: int, num_experts: int,
    ) -> None:
        """Convert Python slot count → byte budget → MOE_INFINITY_SPARSE_BYTES
        so archer's GetSparseCacheLimit returns exactly what Python expects.

        Without this, Python derives cap from sparse_hbm_ratio (YAML) and
        archer derives cache_limit from MOE_INFINITY_SPARSE_RATIO (env) —
        two independent knobs with the same intent that drift if user sets
        one but not the other.

        Also performs PRE-INIT SANITY CHECKS so configuration errors abort
        loudly with a clear message instead of:
          * cudaMalloc OOM inside archer's CPU→GPU SetDevice (silent abort
            via assert(device_memory_ptr != nullptr));
          * 17-minute retry loop in GPUFetchFunc when FindExpertEvict is
            disabled and the budget is undersized.
        """
        # Reuse the exact per-expert byte size resolved in _resolve_capacity
        # (called just before this) so the exported byte budget matches the
        # slot pool archer will allocate.
        expert_bytes = int(getattr(self, "_expert_bytes", 0) or 0)
        if expert_bytes <= 0:
            return
        budget_bytes = int(capacity_per_rank) * int(expert_bytes)

        # _resolve_capacity already clipped cap to HBM × 0.9 ceiling, so by
        # the time we get here budget_bytes is guaranteed to fit.  Defensive
        # re-check: if somehow violated (e.g. caller wired a bare cap), log
        # but still export the byte budget (truth: Python's slot cap drives
        # archer's byte cap; they MUST stay locked).
        try:
            import torch as _torch
            hbm = int(_torch.cuda.get_device_properties(0).total_memory)
        except Exception:
            hbm = 0
        if hbm > 0 and budget_bytes > int(hbm * 0.95):
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[distributed_engine] WARN cap_per_rank "
                      f"({capacity_per_rank} experts × "
                      f"{expert_bytes/1024**2:.1f} MiB = "
                      f"{budget_bytes/1024**3:.2f} GiB) exceeds 95% of "
                      f"HBM — clipping was bypassed somehow.  Forcing "
                      f"MOE_INFINITY_SPARSE_BYTES to HBM×0.85.",
                      flush=True)
            budget_bytes = int(hbm * 0.85)

        # ---- Sanity 2: cap >= per-layer top_k demand ----
        # Otherwise eviction_planner can't find enough victims for a single
        # layer's misses → cache over-runs cap → see eviction_planner WARN.
        top_k = int(getattr(self.config, "num_experts_per_tok", 0) or 0)
        if top_k > 0 and capacity_per_rank < top_k:
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[distributed_engine] WARN cap_per_rank="
                      f"{capacity_per_rank} < num_experts_per_tok={top_k}.  "
                      f"A single layer's miss set will exceed cap on every "
                      f"call.  Raise sparse_hbm_ratio or "
                      f"cache_capacity_per_rank.", flush=True)

        if not os.environ.get("MOE_INFINITY_SPARSE_BYTES"):
            os.environ["MOE_INFINITY_SPARSE_BYTES"] = str(budget_bytes)
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[distributed_engine] MOE_INFINITY_SPARSE_BYTES "
                      f"= {budget_bytes} "
                      f"({budget_bytes/1024**3:.2f} GiB = "
                      f"{capacity_per_rank} experts × "
                      f"{expert_bytes/1024**2:.1f} MiB)", flush=True)

    # ---- controller assembly ----

    def _resolve_capacity(self, num_layers: int, num_experts: int) -> int:
        """Cap selection priority:
          1. explicit cache_capacity_per_rank in yaml (override)
          2. derived from sparse_hbm_ratio + model config (auto)
          3. unlimited (num_layers × num_experts) — last-resort fallback

        The resolved cap is HARD-CLIPPED to fit ``HBM × 0.9`` (Round 5 fix).
        Without this, the unlimited fallback (large models) or a misconfigured
        explicit cache_capacity_per_rank produces a Python cap that's larger
        than archer's actual byte budget — Python keeps installing while
        archer's cache_sizes_ goes negative → cudaMalloc OOM.  The clip
        keeps Python and archer in sync.
        """
        cfg_cap = int((self.ep_cfg or {}).get("cache_capacity_per_rank", 0) or 0)
        ratio = float((self.ep_cfg or {}).get("sparse_hbm_ratio", 0.0) or 0.0)
        # Per-model expert byte size (single source of truth — stored on self
        # so _sync_archer_capacity_env and init_slot_pool reuse the SAME value,
        # eliminating the Python-cap ↔ archer-slot-stride drift).  0 if the
        # model dims are unreadable → fall back to unlimited cap + auto-detect.
        expert_bytes = resolve_expert_bytes(self.config)
        self._expert_bytes = expert_bytes
        if cfg_cap > 0:
            cap = cfg_cap
            src = f"explicit cache_capacity_per_rank={cfg_cap}"
        elif ratio > 0.0 and expert_bytes > 0:
            import torch as _torch
            # HBM is read from the actually-visible CUDA device (varies per
            # node/allocation); cap is per-rank so it's independent of count,
            # but we log device_count for sanity.
            hbm_per_gpu = _torch.cuda.get_device_properties(0).total_memory
            n_visible = _torch.cuda.device_count()
            sparse_budget_bytes = int(hbm_per_gpu * ratio)
            cap = max(1, sparse_budget_bytes // expert_bytes)
            src = (
                f"auto from sparse_hbm_ratio={ratio:.2f} "
                f"(visible CUDA devices={n_visible}, "
                f"HBM/gpu={hbm_per_gpu / 1024**3:.1f}GB): "
                f"{sparse_budget_bytes / 1024**3:.1f} GB / "
                f"{expert_bytes / 1024**2:.0f} MB = cap={cap}"
            )
        else:
            cap = num_layers * num_experts
            src = f"unlimited fallback = {num_layers}×{num_experts}"

        # ---- Hard clip to fit HBM×0.9 (always enforced; the 0.9 envelope must
        #      hold sparse-cache + dense + KV + workspace together) ----
        try:
            import torch as _torch
            hbm = int(_torch.cuda.get_device_properties(0).total_memory)
            if hbm > 0 and expert_bytes > 0:
                # +1 slot: archer reserves one extra slot for staging fetches.
                ceiling_slots = max(1, int(hbm * 0.9) // expert_bytes - 1)
                if cap > ceiling_slots:
                    if int(os.environ.get("RANK", "0")) == 0:
                        print(f"[distributed_engine] CLIP cap {cap} → "
                              f"{ceiling_slots} (HBM×0.9 hard ceiling)",
                              flush=True)
                    cap = ceiling_slots
                    src += f" [clipped to {ceiling_slots}]"
                # Surface how much of the 0.9 envelope is left for dense+KV after
                # the sparse cache, so a too-high sparse_hbm_ratio (cache
                # crowding out dense/KV → OOM) is visible BEFORE the run.
                if int(os.environ.get("RANK", "0")) == 0:
                    cache_gb = (cap + 1) * expert_bytes / 1024**3  # +1 staging
                    leftover_gb = hbm * 0.9 / 1024**3 - cache_gb
                    print(f"[distributed_engine] HBM×0.9 envelope: sparse "
                          f"cache(+staging)={cache_gb:.1f}GB, "
                          f"left for dense+KV={leftover_gb:.1f}GB", flush=True)
        except Exception:
            pass

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[distributed_engine] cap source: {src}", flush=True)
        return cap

    def _build_controller(self, ep_size: int, num_layers: int,
                          num_experts: int, capacity_per_rank: int):
        """Build the Cooperative-Offloading GlobalCacheController.

        2026-05-27 새 architecture:
          * OwnerPolicy (Static/Balanced/DemandAware)        — Phase 1.4
          * RankPlanner with EvictionPolicy (LRU/LFU/DA)     — Phase 2
          * DemandCollector (NCCL all_gather, count-preserving) — Phase 1.1

        CommandDispatcher 는 ``__exit__`` 에서 wired (archer + tensor_map
        의존성).
        """
        from .ep_executor import EPExpertExecutor  # noqa: F401
        from ..controller.global_controller import GlobalCacheController
        from ..controller.owner_policy import build_owner_policy
        from ..controller.evict_policy import build_evict_policy
        from ..controller.rank_planner import RankPlanner
        from ..controller.demand_collector import DemandCollector

        policies_cfg = (self.ep_cfg or {}).get("policies", {}) or {}
        owner_name = policies_cfg.get("fetch_dispatch", "static_placement")
        evict_name = policies_cfg.get("eviction", None)  # None → env default

        # OURS joint placement (baseline #4): fetch_dispatch == "ours" swaps the
        # owner_policy + rank_planner two-step for the joint OursPlacementPlanner.
        ours_planner = None
        demand_accum = None
        rank_accum = None
        retrieval = None
        if owner_name == "ours":
            import os
            import numpy as np
            from ..controller.ours_planner import OursPlacementPlanner
            # evict_cost: "lfu" (freq stub, degenerate quota) / "self" (running
            # accumulated demand) / "retrieval" (history [L,E] cosine top-R).
            evict_cost_mode = os.environ.get("MOE_EP_OURS_EVICT_COST", "lfu")
            ecp = None
            fap = None
            if evict_cost_mode == "self":
                from ..controller.a2_hotness import RunningDemandHotness
                demand_accum = RunningDemandHotness(
                    num_layers, num_experts,
                    mu=float(os.environ.get("MOE_EP_OURS_MU", "0.5")))
                ecp = demand_accum.evict_cost
            elif evict_cost_mode == "retrieval":
                from ..controller.a2_hotness import (
                    RankDemandAccumulator, RetrievalHotnessProvider)
                rank_accum = RankDemandAccumulator(ep_size, num_layers, num_experts)
                retrieval = RetrievalHotnessProvider(
                    ep_size, num_layers, num_experts,
                    topR=int(os.environ.get("MOE_EP_OURS_TOPR", "1")),
                    cap=int(os.environ.get("MOE_EP_OURS_COL_CAP", "1000")))
                cpath = os.environ.get("MOE_EP_OURS_COLLECTION", "")
                if cpath and os.path.exists(cpath):
                    mats = np.load(cpath)            # [N, L, E]
                    retrieval.load_collection([mats[i] for i in range(mats.shape[0])])
                    print(f"[distributed_engine] retrieval collection loaded: "
                          f"{mats.shape} from {cpath}", flush=True)
                ecp = retrieval.evict_cost
                fap = retrieval.future_affinity
            flam = float(os.environ.get(
                "MOE_EP_OURS_LAMBDA", policies_cfg.get("ours_lambda", 0.0)))
            ours_planner = OursPlacementPlanner(
                ep_size=ep_size,
                future_lambda=flam,
                evict_cost_provider=ecp,
                future_affinity_provider=fap,
            )
            owner = build_owner_policy("static_placement")  # unused placeholder
        else:
            owner = build_owner_policy(owner_name)
        evict = build_evict_policy(evict_name)
        rank_planner = RankPlanner(evict_policy=evict)

        controller = GlobalCacheController(
            ep_size=ep_size,
            ep_rank=self.topology.ep_rank,
            cap_per_rank=capacity_per_rank,
            num_layers=num_layers,
            num_experts=num_experts,
            ep_group=self.topology.ep_group,
            owner_policy=owner,
            rank_planner=rank_planner,
            ours_planner=ours_planner,
            demand_accumulator=demand_accum,
            rank_accumulator=rank_accum,
            retrieval_provider=retrieval,
        )
        # MANDATORY for multi-rank: NCCL all_gather of demand.
        controller.demand_collector = DemandCollector(
            ep_size=ep_size, num_experts=num_experts,
            ep_group=self.topology.ep_group,
        )
        print(
            f"[distributed_engine][rank={self.topology.ep_rank}] "
            f"controller built: owner={owner.name} evict={evict.name}",
            flush=True,
        )
        return controller

    def __enter__(self):
        """Run upstream's setup, then swap in our block class.

        Upstream's __enter__ at model_offload.py:330-331 patches
        ``transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock``
        to upstream's ``Qwen3MoEBlock``.  We re-patch to our ``Qwen3MoEBlockEP``
        (which subclasses Qwen3MoEBlock, so the isinstance-based attribute
        injection loop further down still picks up our block).

        We also re-patch ``Qwen3MoeDecoderLayer`` to our subclass so the
        M7 pipeline-overlap path is available when
        ``MOE_EP_PIPELINE_OVERLAP=1`` is set (no-op otherwise).
        """
        rv = super().__enter__()
        from ..models.qwen_ep import Qwen3MoEBlockEP
        from ..models.qwen_decoder_ep import Qwen3MoeDecoderLayerEP
        import transformers.models.qwen3_moe.modeling_qwen3_moe as qmoe
        qmoe.Qwen3MoeSparseMoeBlock = Qwen3MoEBlockEP
        qmoe.Qwen3MoeDecoderLayer = Qwen3MoeDecoderLayerEP
        return rv

    def __exit__(self, exc_type, exc_value, traceback):
        # After upstream tears down (and model is fully built), surface the
        # expert_tensor_map onto our executor + controller.  Upstream's
        # ``expert_tensor_map`` is Dict[(l,e), int] built by OVERWRITING the
        # same key for each of the 3 weight tensors per expert
        # (model_offload.py:573 — gate/up/down).  Only the LAST tensor
        # survives → archer commands would only protect 1/3 of each expert.
        # Rebuild a multi-tensor map from name_id_map directly:
        # Dict[(l,e), List[int]] = all tensor ids per expert.
        exec_ = getattr(self, "expert_executor", None)
        if exec_ is None:
            return super().__exit__(exc_type, exc_value, traceback)
        controller = getattr(exec_, "controller", None)
        try:
            multi_map: dict = self._build_multi_tensor_map()
            if multi_map:
                exec_._expert_tensor_map_cache = multi_map
            elif self.topology.ep_size > 1:
                # Fatal in multi-rank: without a tensor map the controller
                # cannot issue archer commands → drift, no prefetch, no evict.
                raise RuntimeError(
                    "tensor_id map is empty after model load. "
                    "parse_expert_id probably can't match the weight naming "
                    "convention of this model. Check moe_infinity.utils.hf_config."
                )
            # Wire the controller's remaining components now that
            # archer_engine + local_dispatcher + tensor_id_map are populated.
            if controller is not None and multi_map:
                self._wire_controller_components(controller, exec_, multi_map)
                # Hard validation: required components must be wired in multi-rank.
                if self.topology.ep_size > 1:
                    assert controller.command_dispatcher is not None, (
                        "CommandDispatcher not wired — controller can't talk "
                        "to archer. Check archer build (explicit_fetch/evict)."
                    )
        except Exception as e:
            # In single-rank mode an empty map is tolerable (we bypass the
            # controller entirely).  In multi-rank, surface clearly.
            print(f"[distributed_engine.__exit__] wiring failed: {e}",
                  flush=True)
            if self.topology.ep_size > 1:
                # Re-raise so the user sees the failure instead of getting
                # a deadlock at the first cache_sync barrier.
                raise
        return super().__exit__(exc_type, exc_value, traceback)

    def _build_multi_tensor_map(self) -> dict:
        multi_map: dict = {}
        name_id_map = getattr(self, "name_id_map", None)
        config = getattr(self, "config", None)
        if name_id_map and config is not None:
            try:
                from moe_infinity.utils import parse_expert_id
                for name, tid in name_id_map.items():
                    layer_id, expert_id = parse_expert_id(name, config)
                    if expert_id is not None:
                        multi_map.setdefault(
                            (int(layer_id), int(expert_id)), []
                        ).append(int(tid))
            except Exception:
                multi_map = {}
        # Fallback to the legacy single-tensor dict only if the rebuild failed.
        if not multi_map:
            tmap = getattr(self, "expert_tensor_map", None)
            if tmap:
                multi_map = {k: [int(v)] for k, v in tmap.items()}
        return multi_map

    def _wire_controller_components(self, controller, exec_, multi_map: dict) -> None:
        """새 architecture: CommandDispatcher 만 wiring (priority/consistency 제거).

        CommandDispatcher 가 archer 의 ``explicit_evict`` /
        ``explicit_fetch_async`` 페어 (또는 S6 의 ``explicit_replace_async``)
        호출 책임만 가진다.  archer 호출은 controller 에서 launch_fetches 로.
        """
        from ..controller.command_dispatcher import CommandDispatcher

        controller.command_dispatcher = CommandDispatcher(
            archer_engine=getattr(self, "archer_engine", None),
            local_dispatcher=exec_.local_dispatcher,
            rank=controller.ep_rank,
        )

        # 2026-05-28 slot-pool: pre-allocate archer's fixed slot memory once,
        # sized to cap_per_rank slots.  We pass the per-model expert byte size
        # (resolve_expert_bytes, same value cap was derived from) so the slot
        # stride is locked to Python's cap math; 0 falls back to archer's
        # auto-detect (max registered expert) only for unknown models.
        ld = exec_.local_dispatcher
        if ld is not None and hasattr(ld, "init_slot_pool"):
            cap = int(controller.cache_view.cap_per_rank)
            # Pass the SAME per-model expert byte size used to derive cap, so
            # archer's slot stride == Python's cap math (no drift).  0 → archer
            # auto-detects (only when the model dims were unreadable).
            eb = int(getattr(self, "_expert_bytes", 0) or 0)
            ld.init_slot_pool(cap, eb)
            print(f"[distributed_engine][rank={controller.ep_rank}] "
                  f"slot pool init: cap={cap} "
                  f"expert_bytes={'auto' if eb == 0 else eb}", flush=True)

        # EAMC eviction-only (Stage 2): when MOE_EP_EAMC_PRIORITY=1, wrap the
        # (inherited) archer ExpertPredictor in a PriorityAggregator and attach
        # it to the controller.  plan_misses feeds its all_reduced priority
        # matrix to PriorityEviction.  Off by default -> priority_aggregator
        # stays None -> zero overhead, byte-identical LRU/LFU behaviour.
        # EVICTION-ONLY: never prefetches — fetch set is always this layer's
        # misses.
        if os.environ.get("MOE_EP_EAMC_PRIORITY", "0") == "1":
            try:
                from ..controller.priority_aggregator import PriorityAggregator
                predictor = getattr(self, "expert_predictor", None)
                if predictor is not None:
                    controller.priority_aggregator = PriorityAggregator(
                        predictor,
                        num_layers=controller.num_layers,
                        num_experts=controller.num_experts,
                        ep_group=controller.ep_group,
                    )
                    print(f"[distributed_engine][rank={controller.ep_rank}] "
                          f"EAMC priority aggregator attached (eviction-only)",
                          flush=True)
                else:
                    print(f"[distributed_engine][rank={controller.ep_rank}] "
                          f"MOE_EP_EAMC_PRIORITY=1 but no expert_predictor — "
                          f"PriorityEviction falls back to LRU", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[distributed_engine][rank={controller.ep_rank}] "
                      f"EAMC aggregator wiring failed: {e} — LRU fallback",
                      flush=True)
