"""EPExpertExecutor — Controller-Owned Slot Execution runtime (2026-05-29).

Per layer (ep_size>1):
  Phase 1  classify_and_kick_hit   (NCCL all_gather of demand)
  Phase 2  plan_misses             (per-rank deterministic FetchPlan)
  unified  build ONE expert_rank_table (hit owner ∪ miss fetcher)
  fwd a2a  route ALL tokens once  (route_tokens)
  submit   set_inputs(recv) + command_dispatcher.submit_plan(plan, hits)
  compute  archer PlanQueue fetch (direct/staging) + GEMM + combine_partials
           → final_local [K_recv, H]  (wait_layer_done)
  bwd a2a  return_outputs once + scatter_combine_into
  verify   shadow == archer get_cached_experts (drift==0)

Single unified all-to-all (forward) + one return all-to-all (backward): no more
hit/miss 2-pass.  archer does all fetch scheduling + the layer-end combine.

ep_size==1 runs the same controller path without NCCL (recv == local tokens),
so combine_partials directly yields the MoE output.
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional, Set, Tuple

import nvtx
import torch
import torch.nn as nn

from ..exec.nvlink_router import (
    build_expert_rank_table, derive_split_counts, pack_tokens, return_outputs,
    route_tokens,
)
from ..instrument.counters import Counters, _capped_append, _capped_extend
from .routing_forcer import RoutingForcer


ExpertKey = Tuple[int, int]


class EPExpertExecutor:
    EXPERT_BYTES_DEFAULT = 10 * 1024 * 1024  # ~10 MB

    def __init__(
        self,
        archer_engine: Any,
        local_dispatcher: Any,
        topology: Any,
        counters: Any = None,
        expert_bytes: int = EXPERT_BYTES_DEFAULT,
        controller: Any = None,
    ) -> None:
        self.archer_engine = archer_engine
        self.local_dispatcher = local_dispatcher
        self.topology = topology
        self.counters = counters
        self.expert_bytes = expert_bytes
        self.controller = controller
        self._last_layer_id: int = 1 << 30
        self._expert_tensor_map_cache: dict = {}
        self._routing_forcer = RoutingForcer(topology.ep_rank)
        self._heavy_debug: bool = os.environ.get("MOE_EP_HEAVY_DEBUG", "0") == "1"
        # diagnostic: cuda.synchronize at timed-region boundaries so wall-clock
        # phase timers attribute GPU time correctly (default OFF — adds sync).
        self._phase_sync: bool = os.environ.get("MOE_EP_PHASE_SYNC", "0") == "1"
        # Detailed routing/plan/cache trace (env-gated, OFF by default → zero
        # hot-path cost).  When MOE_EP_ROUTING_TRACE=1, every layer dumps one
        # JSONL line to {MOE_EP_ROUTING_TRACE_PATH}/trace_rank{rank}.jsonl with:
        # the global Controller plan (hit_ops + fetch_ops for ALL ranks), the
        # unified expert→GPU routing, per-expert global demand, and the Archer
        # cache slot state (slot→expert per rank) BEFORE and AFTER the layer
        # (= shadow, byte-identical to archer's get_cached_experts, drift=0).
        self._routing_trace_enabled: bool = (
            os.environ.get("MOE_EP_ROUTING_TRACE", "0") == "1"
        )
        self._trace_path: str = os.environ.get(
            "MOE_EP_ROUTING_TRACE_PATH", "/tmp/moe_ep_traces")
        self._trace_fh = None
        self._trace_step: int = -1
        self._trace_last_layer: int = 1 << 30
        # per-phase routed-token accumulators (this rank's Σ K_recv) for the
        # owner-policy benchmark (rank imbalance).  Reset externally per phase.
        self._routed_prefill: int = 0
        self._routed_decode: int = 0
        # per-rank fetch-WORKLOAD accumulators.  The global controller plan
        # assigns each miss a fetcher_rank (naive: expert%ep_size; balanced:
        # least-loaded).  balanced owner-policy balances THIS PCIe fetch load,
        # NOT routed tokens.  plan.fetch_ops is the global plan (identical on
        # every rank) so one rank observes every rank's fetch count.  Reset
        # externally per phase.  Each fetch == one expert (slot_byte_size B).
        self._fetch_ops_rank_prefill: list = []   # [ep] count of fetch_ops/rank
        self._fetch_ops_rank_decode: list = []
        self._miss_modulo_decode: list = []        # [ep] miss experts by e%ep
        self._fetch_layer_imbal_decode: list = []  # capped per-layer max/mean
        # fast-a2a: derive a2a split counts from the controller's global demand
        # matrix (no exchange_counts collective).  Default on.
        self._fast_a2a: bool = os.environ.get("MOE_EP_FAST_A2A", "1") == "1"
        # inter-layer gap (decode only): wall-clock between the END of one
        # run_layer (combine done) and the START of the next (= attention / non-MoE
        # block between consecutive MoE layers).  Reset to 0 on prefill so the
        # prefill->decode boundary is never counted.
        self._last_run_end_ns: int = 0
        self._interlayer_gap_us_decode: float = 0.0
        self._interlayer_gap_n: int = 0

    # Upstream-compatible setters.
    def set_expert_dispatcher(self, dispatcher) -> None:
        self.local_dispatcher = dispatcher

    def set_archer_engine(self, archer_engine) -> None:
        self.archer_engine = archer_engine

    # ============================================================
    # entry point
    # ============================================================
    @nvtx.annotate("EP.run_layer", color="green")
    def run_layer(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        gate: nn.Linear,
        lib: Any,
        is_decode: bool,
        batch_size: Any = None,
        sequence_length: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # inter-layer gap (decode only): time from the previous run_layer end to
        # this start = the attention/non-MoE block between consecutive MoE layers.
        if is_decode:
            if self._last_run_end_ns:
                self._interlayer_gap_us_decode += (
                    time.perf_counter_ns() - self._last_run_end_ns) / 1000.0
                self._interlayer_gap_n += 1
        else:
            self._last_run_end_ns = 0  # don't count the prefill->decode boundary

        router_logits = gate(hidden_states)
        router_mask, routing_weights_mask = lib.topk_softmax(router_logits)
        # routing record/force (OFF unless env set) -> fair cross-policy demand.
        router_mask, routing_weights_mask = self._routing_forcer.apply(
            router_mask, routing_weights_mask)
        self._current_batch_shape = (batch_size, sequence_length)

        if self.counters is not None and layer_id <= self._last_layer_id:
            self.counters.advance_step()
        self._last_layer_id = layer_id

        if self._heavy_debug:
            print(f"[exec rank={self.topology.ep_rank}] L{layer_id} "
                  f"is_decode={is_decode} hidden={tuple(hidden_states.shape)} "
                  f"dtype={hidden_states.dtype} "
                  f"router_mask={tuple(router_mask.shape)}", flush=True)

        if self.topology.ep_size == 1:
            final = self._single_rank_dispatch(
                layer_id, hidden_states, router_mask,
                routing_weights_mask, is_decode)
        else:
            final = self._coop_dispatch(
                layer_id, hidden_states, router_mask,
                routing_weights_mask, is_decode)
        # stamp end (combine done) so the next run_layer can measure the gap.
        self._last_run_end_ns = time.perf_counter_ns()
        return final, router_logits

    # ============================================================
    # ep_size==1 — same controller path, no NCCL.
    # ============================================================
    def _single_rank_dispatch(
        self, layer_id, hidden_states, router_mask, routing_weights_mask,
        is_decode,
    ) -> torch.Tensor:
        assert self.controller is not None, (
            "single-rank coslot path requires a controller (build it for "
            "ep_size>=1 in distributed_engine).")
        self.controller.begin_layer(layer_id)
        p1 = self.controller.classify_and_kick_hit(layer_id, router_mask)
        # EAMC eviction-only hook: no-op unless a priority_aggregator is attached.
        self.controller.maybe_inject_eamc_priority(
            layer_id, router_mask=router_mask,
            seq_ids=getattr(self, "_current_seq_ids", None))
        plan = self.controller.plan_misses(p1, layer_id, is_decode=is_decode)
        # recv == local tokens (multi-hot router_mask: each token picks top_k).
        self.local_dispatcher.set_inputs(
            hidden_states, router_mask.bool(), routing_weights_mask,
            is_decode=is_decode)
        self.controller.command_dispatcher.submit_plan(
            plan, p1.hit_launch_info.hit_ops, 0)
        final_local = self.local_dispatcher.wait_layer_done()
        dtype = hidden_states.dtype
        if final_local.dtype != dtype:
            raise RuntimeError(
                f"[exec] single-rank final_local dtype {final_local.dtype} != "
                f"model {dtype} (L{layer_id})")
        self.controller.verify_layer_end(layer_id)
        return final_local

    # ============================================================
    # ep_size>1 — unified single forward/backward all-to-all.
    # ============================================================
    @nvtx.annotate("EP.coop_dispatch", color="orange")
    def _coop_dispatch(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        router_mask: torch.Tensor,
        routing_weights_mask: torch.Tensor,
        is_decode: bool,
    ) -> torch.Tensor:
        assert self.controller is not None, "ep_size>1 requires controller"
        topo = self.topology
        ep_size = topo.ep_size
        ep_group = topo.ep_group
        device = hidden_states.device
        dtype = hidden_states.dtype
        N, H = hidden_states.shape
        num_experts = router_mask.size(1)
        counters: Counters = self.counters  # type: ignore[assignment]
        layer_t0 = time.perf_counter_ns() if counters is not None else 0

        # ----- Phase 0 -----
        self.controller.begin_layer(layer_id)

        # ----- Phase 1: classify (global demand all_gather) -----
        if counters is not None:
            with counters.time("cache_sync_us"), \
                    counters.time_phased("cache_sync_us", is_decode):
                p1 = self.controller.classify_and_kick_hit(layer_id, router_mask)
        else:
            p1 = self.controller.classify_and_kick_hit(layer_id, router_mask)

        # ----- routing trace: snapshot cache slots BEFORE plan mutates -----
        slots_before = None
        if self._routing_trace_enabled:
            slots_before = [
                list(self.controller.cache_view.per_rank[r].slots)
                for r in range(ep_size)
            ]

        # EAMC eviction-only hook: no-op unless a priority_aggregator is attached
        # (must run on every rank — aggregator all_reduce is collective).
        self.controller.maybe_inject_eamc_priority(
            layer_id, router_mask=router_mask,
            seq_ids=getattr(self, "_current_seq_ids", None))

        # ----- Phase 2: per-rank miss plan (shadow mutate) -----
        plan = self.controller.plan_misses(p1, layer_id, is_decode=is_decode)
        # Unified routing table: hit owner ∪ miss fetcher (disjoint, asserted
        # in plan_misses).
        expert_to_rank = plan.expert_to_rank

        table = build_expert_rank_table(
            expert_to_rank, layer_id, num_experts, device)

        # fast-a2a split counts (no exchange_counts collective).
        send_pre = recv_pre = None
        if self._fast_a2a and p1.per_rank_count is not None:
            prl = p1.per_rank_count.tolist()
            send_pre, recv_pre = derive_split_counts(
                prl, expert_to_rank, layer_id, topo.ep_rank, ep_size)

        # ----- forward all-to-all: route ALL demanded tokens once -----
        with self._time_or_noop("a2a_forward_us", is_decode):
            rplan = pack_tokens(
                hidden_states, router_mask, routing_weights_mask, table,
                ep_size, precomp_send_counts=send_pre)
            recv_hidden, recv_expert_idx, recv_weights, recv_counts = \
                route_tokens(rplan, ep_group, layer_id=layer_id,
                             recv_counts=recv_pre)

        K_recv = recv_hidden.size(0)
        # routed-token accounting (this rank's processed tokens this layer).
        if is_decode:
            self._routed_decode += int(K_recv)
        else:
            self._routed_prefill += int(K_recv)
        # per-rank fetch-workload accounting (global plan → every rank's count).
        acc = (self._fetch_ops_rank_decode if is_decode
               else self._fetch_ops_rank_prefill)
        if len(acc) != ep_size:
            acc[:] = [0] * ep_size
        layer_counts = [0] * ep_size
        for op in plan.fetch_ops:
            acc[op.fetcher_rank] += 1
            layer_counts[op.fetcher_rank] += 1
        if is_decode:
            if len(self._miss_modulo_decode) != ep_size:
                self._miss_modulo_decode[:] = [0] * ep_size
            for op in plan.fetch_ops:
                self._miss_modulo_decode[op.expert[1] % ep_size] += 1
            tot = sum(layer_counts)
            if tot > 0:
                _capped_append(self._fetch_layer_imbal_decode,
                               max(layer_counts) / (tot / ep_size))
        # one-hot recv router mask + weights for archer set_inputs (each recv
        # row is exactly one (token, expert) pair).
        recv_router_mask = torch.zeros(
            K_recv, num_experts, dtype=torch.bool, device=device)
        recv_router_weights = torch.zeros(
            K_recv, num_experts, dtype=recv_weights.dtype, device=device)
        if K_recv > 0:
            recv_router_mask.scatter_(1, recv_expert_idx.unsqueeze(1), True)
            recv_router_weights.scatter_(
                1, recv_expert_idx.unsqueeze(1), recv_weights.unsqueeze(1))
        self.local_dispatcher.set_inputs(
            recv_hidden, recv_router_mask, recv_router_weights,
            is_decode=is_decode)

        # ----- submit plan + run (archer fetch + GEMM + combine) -----
        self.controller.command_dispatcher.submit_plan(
            plan, p1.hit_launch_info.hit_ops, topo.ep_rank)
        with self._time_or_noop("local_exec_us", is_decode):
            final_local = self.local_dispatcher.wait_layer_done()

        # routing trace: capture archer's ACTUAL C++ residency (get_cached_experts)
        # right after it executed this layer's plan — this is the GROUND TRUTH
        # the shadow is verified against (NOT the Python shadow).
        archer_after = None
        if self._routing_trace_enabled:
            try:
                archer_after = {
                    (int(l), int(e))
                    for (l, e) in self.local_dispatcher.get_cached_experts(0)
                }
            except Exception:
                archer_after = set()

        # Fail-fast #9: dtype must match model before the a2a.
        if final_local.dtype != dtype:
            raise RuntimeError(
                f"[exec rank={topo.ep_rank}] L{layer_id} final_local dtype "
                f"{final_local.dtype} != model {dtype} before a2a")

        # ----- backward all-to-all + scatter -----
        final = torch.zeros(N, H, dtype=dtype, device=device)
        with self._time_or_noop("a2a_backward_us", is_decode):
            send_back = return_outputs(
                final_local, recv_counts, rplan.send_counts, ep_group,
                layer_id=layer_id)
        scatter_combine_into(final, send_back, rplan.token_idx, dtype)

        # ----- Phase 4: verify (drift==0 or raise) -----
        drift_count, _, _ = self.controller.verify_layer_end(layer_id)

        if self._routing_trace_enabled:
            self._record_trace(layer_id, is_decode, p1, plan, slots_before,
                               archer_after)

        # ----- counters -----
        if counters is not None:
            n_hits = p1.n_hits
            n_misses = p1.n_misses_dedup
            n_my_fetch = sum(
                1 for op in plan.fetch_ops if op.fetcher_rank == topo.ep_rank)
            counters.cache_hits += n_hits
            counters.cache_misses += n_misses
            counters.cache_evictions += sum(
                1 for op in plan.fetch_ops
                if op.fetcher_rank == topo.ep_rank
                and op.victim_expert is not None)
            _capped_append(counters.layer_hits, n_hits)
            _capped_append(counters.layer_misses, n_misses)
            _capped_append(counters.drift_layer_pre, int(drift_count))
            counters.layers_executed += 1
            counters.tokens_processed += int(N)
            layer_total_us = int((time.perf_counter_ns() - layer_t0) // 1000)
            _capped_append(counters.layer_total_us, layer_total_us)
            if is_decode:
                _capped_append(counters.decode_layer_total_us, layer_total_us)
                counters.decode_tokens_processed += int(N)
                counters.decode_cache_hits += n_hits
                counters.decode_cache_misses += n_misses
            else:
                _capped_append(counters.prefill_layer_total_us, layer_total_us)
                counters.prefill_tokens_processed += int(N)
                counters.prefill_cache_hits += n_hits
                counters.prefill_cache_misses += n_misses
            counters.pcie_fetch_bytes += int(n_my_fetch) * int(self.expert_bytes)
            cd = self.controller.command_dispatcher
            if cd is not None:
                counters.prefetch_issued_count = cd.n_fetches
                counters.dispatch_fetch_failures = cd.n_fetch_failures
                counters.dispatch_evict_failures = cd.n_evict_failures
                counters.dispatch_evict_noops = cd.n_evict_noops
                try:
                    fetch_us, evict_us = cd.drain_latency()
                    _capped_extend(counters.dispatch_fetch_us, fetch_us)
                    _capped_extend(counters.dispatch_evict_us, evict_us)
                except Exception:
                    pass
        return final

    # ============================================================
    # helpers
    # ============================================================
    def _time_or_noop(self, key: str, is_decode: bool):
        if self.counters is None:
            from contextlib import nullcontext
            return nullcontext()
        cm = _StackedCM(
            self.counters.time(key),
            self.counters.time_phased(key, is_decode),
        )
        if self._phase_sync:
            return _PhaseSyncCM(cm)
        return cm

    # ------------------------------------------------------------
    # detailed routing/plan/cache trace (env-gated)
    # ------------------------------------------------------------
    def _record_trace(self, layer_id, is_decode, p1, plan, slots_before,
                      archer_after):
        """Dump one JSONL line per layer with (a) the Controller plan + unified
        routing, (b) the shadow slot state before/after, (c) the ACTUAL archer
        C++ residency (get_cached_experts) vs the shadow for drift, and (d) a
        from-scratch HIT VERIFICATION: demanded ∩ resident-before computed
        independently of classify, so true_hit_rate (intersection) can be
        compared against submit_hit_rate (HitOps) — if they match, hits=0 is
        real, not a classify bug."""
        import json
        from collections import Counter as _C
        topo = self.topology
        ep_size = topo.ep_size
        if layer_id <= self._trace_last_layer:
            self._trace_step += 1
        self._trace_last_layer = layer_id

        if self._trace_fh is None:
            import os as _os
            _os.makedirs(self._trace_path, exist_ok=True)
            rid = os.environ.get("MOE_EP_RUN_TAG", "trace")
            self._trace_fh = open(
                f"{self._trace_path}/{rid}_routing_rank{topo.ep_rank}.jsonl",
                "w", buffering=1)

        def _k(c):           # (l,e) → "l:e"
            return f"{c[0]}:{c[1]}"

        def _slot_str(cells):
            return [None if c is None else _k(c) for c in cells]

        # ---- HIT VERIFICATION (independent of classify) ----
        demanded = set(p1.demand.keys())                 # {(layer_id, e)} this layer
        resident_before = [
            {c for c in slots_before[r] if c is not None} for r in range(ep_size)
        ]
        resident_union = set().union(*resident_before) if resident_before else set()
        global_inter = demanded & resident_union          # true global hits
        local_inter = [sorted(_k(x) for x in (demanded & resident_before[r]))
                       for r in range(ep_size)]
        hit_keys = {op.expert for op in p1.hit_launch_info.hit_ops}
        miss_keys = {op.expert for op in plan.fetch_ops}
        # cross-check: classify's hit set should equal the independent global
        # intersection.  Discrepancy = classify bug (logged for the summary).
        classify_matches_intersection = (hit_keys == global_inter)

        cache_before_layer_dist = dict(_C(c[0] for c in resident_union))

        # ---- shadow (this rank, end-of-layer) vs ACTUAL archer C++ residency ----
        shadow_after_set = self.controller.cache_view.per_rank[topo.ep_rank].occupied_keys()
        archer_set = archer_after if archer_after is not None else set()
        cache_match = (shadow_after_set == archer_set)
        drift_missing = sorted(_k(x) for x in (shadow_after_set - archer_set))[:10]
        drift_extra = sorted(_k(x) for x in (archer_set - shadow_after_set))[:10]

        shadow_after = [_slot_str(self.controller.cache_view.per_rank[r].slots)
                        for r in range(ep_size)]
        cache_before = [_slot_str(s) for s in slots_before]
        updates = []
        for r in range(ep_size):
            for s in range(len(shadow_after[r])):
                if cache_before[r][s] != shadow_after[r][s]:
                    updates.append({"rank": r, "slot": s,
                                    "evict": cache_before[r][s],
                                    "install": shadow_after[r][s]})

        fetch_ops = [{"expert": _k(op.expert), "fetcher_gpu": op.fetcher_rank,
                      "dst_slot": op.dst_slot,
                      "victim": (None if op.victim_expert is None
                                 else _k(op.victim_expert)),
                      "order": op.order}
                     for op in plan.fetch_ops]
        hit_ops = [{"expert": _k(op.expert), "serving_gpu": op.owner_rank,
                    "slot": op.slot} for op in p1.hit_launch_info.hit_ops]

        rec = {
            "step": self._trace_step,
            "phase": "decode" if is_decode else "prefill",
            "layer": layer_id,
            "ep_size": ep_size,
            # --- hit verification (item: is hit=0 real?) ---
            "demanded_count": len(demanded),
            "resident_before_total": sum(len(s) for s in resident_before),
            "resident_before_per_rank": [len(s) for s in resident_before],
            "global_intersection_count": len(global_inter),
            "global_intersection_sample": sorted(_k(x) for x in global_inter)[:5],
            "local_intersection_count_by_rank": [len(demanded & resident_before[r])
                                                 for r in range(ep_size)],
            "hit_ops_count": len(hit_keys),
            "miss_ops_count": len(miss_keys),
            "true_global_hits": len(global_inter),
            "submit_hits": len(hit_keys),
            "classify_matches_intersection": classify_matches_intersection,
            "hit_sample": sorted(_k(x) for x in hit_keys)[:5],
            "miss_sample": sorted(_k(x) for x in miss_keys)[:5],
            "cache_before_layer_dist": cache_before_layer_dist,
            # --- routing + plan ---
            "demand": {_k(k): c for k, c in p1.demand.items()},
            "routing": {_k(k): r for k, r in plan.expert_to_rank.items()},
            "controller_plan": {"hit_ops": hit_ops, "fetch_ops": fetch_ops},
            # --- shadow vs ACTUAL archer C++ residency (item 1,4) ---
            "shadow_cache_after": shadow_after,
            "archer_get_cached_experts": sorted(_k(x) for x in archer_set),
            "shadow_resident_this_rank": sorted(_k(x) for x in shadow_after_set),
            "cache_match_shadow_vs_archer": cache_match,
            "drift_missing_in_archer": drift_missing,
            "drift_extra_in_archer": drift_extra,
            "archer_slot_updates": updates,
        }
        self._trace_fh.write(json.dumps(rec) + "\n")


class _StackedCM:
    """두 context manager 를 한 with 로 감싸는 helper."""
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __enter__(self):
        self.a.__enter__()
        self.b.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.b.__exit__(exc_type, exc, tb)
        self.a.__exit__(exc_type, exc, tb)


class _PhaseSyncCM:
    """MOE_EP_PHASE_SYNC=1 diagnostic wrapper: cuda.synchronize at BOTH region
    boundaries.  entry-sync drains prior-stream residue (excluded from this
    timer); exit-sync forces this region's own GPU work to finish inside the
    timer → wall-clock phase attribution becomes truthful.  Measurement-only
    (adds sync stalls) — never enable when collecting perf numbers."""
    def __init__(self, inner):
        self.inner = inner
    def __enter__(self):
        torch.cuda.synchronize()
        self.inner.__enter__()
        return self
    def __exit__(self, exc_type, exc, tb):
        torch.cuda.synchronize()
        return self.inner.__exit__(exc_type, exc, tb)


def scatter_combine_into(
    target: torch.Tensor,     # [N, H], 누적 대상 (in-place)
    send_back: torch.Tensor,  # [K, H], a2a 로 돌아온 GEMM 결과
    token_idx: torch.Tensor,  # [K]
    dtype: torch.dtype,
) -> None:
    """combine 결과를 token 위치로 scatter-add (in-place)."""
    if send_back.numel() == 0:
        return
    target.index_add_(0, token_idx, send_back.to(dtype))
