"""GlobalCacheController — Cooperative offloading single source of truth.

Design 참조: /home/work/hyewon.lee/실험/main_exp/moe_cooperative_offloading_design.md

Phase 분리 (ep_executor 가 사이에 비동기 작업 끼움):
  Phase 0 begin_layer            — 직전 layer 끝 verify (optional)
  Phase 1 classify_and_kick_hit  — global demand → union dedup miss →
                                   hit early-kick (hit a2a launch 위해 hit list 반환)
                                   → owner_policy.assign_rank (miss → rank 분배)
  Phase 2 plan_misses             — per-rank sequential plan (shadow 즉시 mutate)
                                   → LayerPlan (fetch_ops + routing_map_miss)
  Phase 4 verify_layer_end        — shadow == archer_physical assert

핵심 invariant:
  I1. python_shadow == archer_physical  (Phase 0 보장, Phase 4 assert)
  I2. shadow mutate 는 controller 만   (rank_planner.apply 만)
  I3. shadow 갱신은 Archer 비대기       (Phase 2 끝나면 shadow = layer 종료 후 상태)
  I5. hit-compute read 가 fetch-replace write 보다 항상 먼저 끝남
      (archer 측 compute_event wait — S6 에서 명시적 보장)
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

import torch

from .layer_plan import (
    ExpertKey, FetchOp, HitLaunchInfo, HitOp, LayerPlan, Phase1Output,
)
from .slot_cache import GlobalSlotCacheView, RankSlotCache


# Phase 4 verify 활성/비활성 (default ON — drift 는 silent corruption 의
# 가장 빠른 detection).  안정화 후 sampling 또는 OFF 가능.
_VERIFY_LAYER_END = os.environ.get("MOE_EP_VERIFY_LAYER_END", "1") == "1"
# Heavy debug — Phase 별 상세 print.
_HEAVY = os.environ.get("MOE_EP_HEAVY_DEBUG", "0") == "1"
# Placement diagnostics — per-layer NVLink(local/remote token) + victim-in-demand
# accounting.  OFF by default (zero overhead); the placement experiment turns it
# on via MOE_EP_PLACEMENT_STATS=1 to attribute WHERE OURS wins.
_PLACEMENT_STATS = os.environ.get("MOE_EP_PLACEMENT_STATS", "0") == "1"


def _new_placement_bucket() -> dict:
    return {
        "layers": 0, "demand_tokens": 0,
        "local_tokens": 0, "remote_tokens": 0,          # all demanded (hit∪miss) → a2a
        "miss_local_tokens": 0, "miss_remote_tokens": 0,  # miss-only (what placement decides)
        "hit_tokens": 0, "miss_tokens": 0,
        "n_fetch": 0, "victim_in_demand": 0,            # victim is a THIS-layer hit (→ staging)
        "victim_empty": 0, "victim_cold": 0,
        "n_hits": 0, "n_miss": 0,
        # bottleneck fetch: per-(step,layer) MAX-over-ranks fetch count, summed.
        # wall-clock per layer is bounded by the busiest rank's fetch round, so
        # this = total bottleneck fetch rounds (quota DP should lower it).
        "sum_max_fetch": 0, "sum_mean_fetch": 0.0,
        # diagnostics for "why random≫balanced at same cache/LFU":
        #   hits_by_rank   : per-rank hit count (dead-rank check)
        #   occ_slots_sum  : Σ total_slots_used over sampled layers
        #   occ_unique_sum : Σ global distinct-resident experts (slots_used-unique
        #                    = duplicate copies wasting capacity)
        #   occ_per_rank_sum: per-rank occupancy sum (balance of cache fill)
        "hits_by_rank": {}, "occ_samples": 0, "occ_slots_sum": 0,
        "occ_unique_sum": 0, "occ_per_rank_sum": {},
        "by_layer": {},  # layer_id -> [n_fetch, victim_in_demand, remote_tok, demand_tok]
    }


class GlobalCacheController:
    """Cooperative offloading controller.

    Owns:
      * cache_view (GlobalSlotCacheView) — single source of truth
      * owner_policy                     — Phase 1.4 rank 분배
      * rank_planner                     — Phase 2 per-rank sequential plan
      * demand_collector                 — Phase 1.1 NCCL all_gather
      * command_dispatcher               — Archer bridge (wired in __exit__)
    """

    def __init__(
        self,
        ep_size: int,
        ep_rank: int,
        cap_per_rank: int,
        num_layers: int,
        num_experts: int,
        ep_group,
        owner_policy,
        rank_planner,
        demand_collector=None,
        command_dispatcher=None,
        ours_planner=None,
        demand_accumulator=None,
        rank_accumulator=None,
        retrieval_provider=None,
    ):
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.ep_group = ep_group

        self.cache_view = GlobalSlotCacheView(
            ep_size=ep_size, cap_per_rank=cap_per_rank,
            num_layers=num_layers, num_experts=num_experts,
        )
        self.owner_policy = owner_policy
        self.rank_planner = rank_planner
        self.demand_collector = demand_collector
        self.command_dispatcher = command_dispatcher
        # OURS joint placement (baseline #4).  When set, it REPLACES the
        # owner_policy (Phase 1.4) + per-rank rank_planner (Phase 2) two-step:
        # ownership + slot + fetch order are decided jointly in plan_misses.
        # None (default) -> baselines run the unchanged two-step path, so every
        # existing call site behaves byte-identically.
        self._ours_planner = ours_planner
        self._ours_mode = ours_planner is not None
        # Optional self-[L,E] hotness: accumulates the (all-gathered, hence
        # rank-identical) per-layer demand so OURS evict_cost reflects "demanded a
        # lot so far -> hot".  None -> OURS uses LFU-freq stub.  Updated in
        # classify_and_kick_hit (deterministic -> drift-safe).
        self._demand_accum = demand_accumulator
        # retrieval-based hotness (C): per-rank accumulator + history retrieval.
        # rank_accum updated each layer from all-gathered per_rank_count; retrieval
        # refreshed per decode step (layer 0).  Both deterministic -> drift-safe.
        self._rank_accum = rank_accumulator
        self._retrieval = retrieval_provider
        # churn 계측: (l,e)별 fetch 횟수(=버렸다 다시 사온 횟수) + rank별 고유 expert.
        # churn_factor = total_fetches/distinct = expert당 평균 재-fetch (thrash 지표).
        self._churn_ctr = {"prefill": {}, "decode": {}}
        self._churn_rankset = {"prefill": [set() for _ in range(ep_size)],
                               "decode": [set() for _ in range(ep_size)]}

        # Placement diagnostics (filled only when _PLACEMENT_STATS).
        self.placement_stats = {
            "prefill": _new_placement_bucket(),
            "decode": _new_placement_bucket(),
        }
        # EAMC: optional [num_layers, num_experts] priority matrix injected per
        # layer (by ep_executor from the priority_aggregator) and consumed by an
        # ``eamc`` evict_policy.  None -> evict_policy falls back (LRU).  Only
        # PriorityEviction reads it; LRU/LFU/DemandAware ignore it.
        #
        # SCOPE: EAMC here is EVICTION-ONLY (which resident expert to drop).  It
        # does NOT prefetch — the fetch set is always just this layer's misses
        # (owner_policy); no future-layer expert is ever pulled in.
        self._layer_priority = None
        # Optional PriorityAggregator (set by the integration when EAMC priority
        # is enabled).  None by default -> maybe_inject_eamc_priority is a no-op,
        # so normal LRU/LFU runs see zero overhead and byte-identical behaviour.
        self.priority_aggregator = None

        # Stats
        self._n_drift_events = 0
        self._cumulative_drift = 0
        self._n_unconditional_drift_logs = max(
            0, int(os.environ.get("MOE_EP_DRIFT_LOG_FIRST_N", "20") or 20))

    # ============================================================
    # Phase 0 — Layer 진입 (no-op; verify 는 Phase 4 가 직전 layer 의 끝에서)
    # ============================================================
    def begin_layer(self, layer_id: int) -> None:
        if _HEAVY:
            print(f"[ctrl rank={self.ep_rank}] L{layer_id} BEGIN "
                  f"shadow_total={self.cache_view.total_slots_used()}",
                  flush=True)

    # ============================================================
    # Phase 1 — Global classification + hit early-kick
    # ============================================================
    def classify_and_kick_hit(
        self,
        layer_id: int,
        local_router_mask: torch.Tensor,
    ) -> Phase1Output:
        """Phase 1.1-1.4 통합 실행.  반환 즉시 ep_executor 가 hit a2a launch.

        local_router_mask: [N_local, num_experts] bool, this rank 의 router 결과.
        """
        # ----- Phase 1.1: global demand -----
        if self.demand_collector is None or self.ep_size <= 1:
            # single-rank or test: stub
            per_rank = self._stub_per_rank_count(local_router_mask)
        else:
            per_rank = self.demand_collector.collect_with_count(
                layer_id, local_router_mask)
        #   per_rank: [ep_size, num_experts] int64 (cpu)
        union_bool = (per_rank.sum(dim=0) > 0)
        demanded_ids: List[int] = union_bool.nonzero(as_tuple=True)[0].tolist()
        global_count = per_rank.sum(dim=0).tolist()  # [num_experts]

        # demand[(layer, expert_id)] = token count (이 layer 한정)
        demand: Dict[ExpertKey, int] = {
            (layer_id, e): int(global_count[e])
            for e in demanded_ids
        }
        # self-[L,E] hotness: accumulate this layer's (rank-identical) demand.
        if self._demand_accum is not None:
            self._demand_accum.update(demand)
        # retrieval hotness: accumulate per-rank demand [G,E] for this layer.
        if self._rank_accum is not None:
            self._rank_accum.update(layer_id, per_rank.cpu().numpy())

        # ----- Phase 1.2: hit / miss with union dedup -----
        hit_set: Set[ExpertKey] = set()
        miss_set: Set[ExpertKey] = set()
        hit_ops_list: List[HitOp] = []
        routing_map_hit: Dict[ExpertKey, int] = {}

        for e in demanded_ids:
            key = (layer_id, e)
            owner = self.cache_view.locate(key)
            if owner is None:
                miss_set.add(key)
            else:
                hit_set.add(key)
                slot = self.cache_view.per_rank[owner].find_slot(key)
                assert slot is not None, (
                    f"locate={owner} but find_slot=None for {key}")
                hit_ops_list.append(HitOp(
                    expert=key, owner_rank=owner, slot=slot))
                routing_map_hit[key] = owner

        # ----- Phase 1.3: touch hits (meta.last_used 갱신) -----
        # 새 design 의 I5 가정 하에 protected_keys 는 불필요.
        # touch 는 meta 갱신만 (eviction policy 가 fresh state 보도록).
        for op in hit_ops_list:
            self.cache_view.per_rank[op.owner_rank].touch_use(
                op.expert, demand_count=demand.get(op.expert, 0))

        # ----- Phase 1.4: owner assignment (union dedup miss) -----
        miss_sorted = sorted(miss_set)
        if self._ours_mode:
            # OURS decides ownership jointly with slot/order in plan_misses.
            # Leave miss_per_rank empty; the planner reads miss_set instead.
            miss_per_rank: Dict[int, List[ExpertKey]] = {
                r: [] for r in range(self.ep_size)}
        else:
            miss_per_rank = self.owner_policy.assign_rank(
                miss_sorted, demand, self.cache_view,
            )

        if _HEAVY:
            print(f"[ctrl rank={self.ep_rank}] L{layer_id} P1: "
                  f"demanded={len(demanded_ids)} hits={len(hit_set)} "
                  f"miss_dedup={len(miss_set)} "
                  f"miss_per_rank={ {r: len(v) for r, v in miss_per_rank.items()} }",
                  flush=True)

        return Phase1Output(
            hit_launch_info=HitLaunchInfo(
                layer_id=layer_id,
                hit_ops=hit_ops_list,
                routing_map_hit=routing_map_hit,
            ),
            miss_per_rank=miss_per_rank,
            demand=demand,
            miss_set=miss_sorted,
            n_unique_demand=len(demanded_ids),
            n_hits=len(hit_set),
            n_misses_dedup=len(miss_set),
            # fast-a2a: hand the (CPU) demand-count matrix to the executor so it
            # can derive a2a split sizes without a per-route exchange_counts.
            per_rank_count=per_rank,
        )

    # ============================================================
    # Phase 2 — Per-rank sequential plan (shadow 즉시 mutate)
    # ============================================================
    def set_layer_priority(self, priority) -> None:
        """Inject this layer's EAMC priority matrix (or None to clear).

        Called by ep_executor before ``plan_misses`` when MOE_EP_EAMC_PRIORITY
        is on.  Must be byte-identical across ranks (priority_aggregator
        all_reduce guarantees this) so every rank's plan stays deterministic.
        """
        self._layer_priority = priority

    def maybe_inject_eamc_priority(self, layer_id: int, **kwargs) -> None:
        """EVICTION-ONLY EAMC hook (no prefetch).

        No-op unless a ``priority_aggregator`` is attached.  When present and
        enabled, computes this layer's priority matrix (NCCL all_reduce inside
        the aggregator -> byte-identical across ranks) and stores it for
        ``plan_misses`` to feed ``PriorityEviction``.  Best-effort: any failure
        falls back to ``None`` (-> LRU), never crashes the layer.  Does NOT add
        any fetch — only influences which resident expert is evicted.

        DEADLOCK NOTE: when an aggregator is attached, EVERY rank must reach
        this call each layer (the aggregator's all_reduce is collective); the
        early ``return`` below only triggers when NO aggregator is attached,
        which is an env/config decision identical on every rank.
        """
        agg = self.priority_aggregator
        if agg is None:
            return
        router_mask = kwargs.get("router_mask")
        seq_ids = kwargs.get("seq_ids")
        # Derive per-seq selected-expert ids (best-effort).  Any failure -> None;
        # aggregate is STILL called so the collective all_reduce stays in
        # lockstep across ranks (deadlock-safe).
        try:
            expert_index = self._router_mask_to_expert_index(router_mask, seq_ids)
        except Exception:
            expert_index = None
        try:
            matrix = agg.aggregate(
                seq_ids=tuple(seq_ids) if seq_ids else (),
                layer_id=layer_id,
                expert_index_per_seq=expert_index,
            )
        except Exception:
            matrix = None
        self._layer_priority = matrix

    @staticmethod
    def _router_mask_to_expert_index(router_mask, seq_ids):
        """Per-seq selected-expert ids from a [N, num_experts] router mask.

        Returns a list (len == len(seq_ids)) of 1-D numpy arrays of expert ids,
        or None if it can't be derived.  Used only to feed the EAMC tracer
        (eviction signal); never affects fetch.
        """
        if router_mask is None or not seq_ids:
            return None
        rm = router_mask.reshape(-1, router_mask.shape[-1]) != 0   # [N, E] bool
        n = int(rm.shape[0])
        b = len(seq_ids)

        def cols(sub):
            return sub.nonzero(as_tuple=True)[-1].detach().cpu().numpy()

        if b == n:
            return [cols(rm[i:i + 1]) for i in range(n)]
        if b == 1:
            return [cols(rm)]
        if n % b == 0:
            per = n // b
            return [cols(rm[i * per:(i + 1) * per]) for i in range(b)]
        return None

    def plan_misses(
        self,
        phase1: Phase1Output,
        layer_id: int,
        priority=None,
        is_decode: bool = False,
    ) -> LayerPlan:
        """모든 rank 의 LayerPlan 을 같은 controller 인스턴스가 produce.

        deterministic 보장:
          * owner_policy.assign_rank   — 모든 rank 같은 입력 → 같은 출력
          * rank_planner.plan          — demand 정렬 + sequential apply, 결정성
        → 모든 rank 의 shadow 가 byte-identical mutate.

        모든 rank 의 fetch_ops 를 한 plan 으로 모음 (controller 가 다른 rank
        의 의도까지 알아야 token routing 결정 + verify 가능).  그러나 실제
        archer 명령은 자기 rank op 만 issue (CommandDispatcher 에서 filter).
        """
        # EAMC priority: explicit arg wins, else the per-layer injected matrix.
        eff_priority = priority if priority is not None else self._layer_priority

        # retrieval hotness: refresh once per decode STEP (layer 0) from the
        # per-rank accumulator (all-gathered -> identical on every rank -> the
        # resulting evict_cost/future_affinity are byte-identical -> drift-safe).
        if (self._retrieval is not None and self._rank_accum is not None
                and is_decode and layer_id == 0):
            self._retrieval.refresh(self._rank_accum.all())

        all_fetch_ops: List[FetchOp]
        if self._ours_mode:
            # OURS joint placement: ownership + slot + order decided together.
            # Mutates the replicated shadow via rank_cache.apply (same contract
            # as RankPlanner) so Phase-4 verify stays drift=0.
            all_fetch_ops = self._ours_planner.plan_layer(
                miss_set=phase1.miss_set,
                demand=phase1.demand,
                per_rank_count=phase1.per_rank_count,
                cache_view=self.cache_view,
                layer_id=layer_id,
                is_decode=is_decode,
            )
        else:
            all_fetch_ops = []
            for r in range(self.ep_size):
                misses_r = phase1.miss_per_rank.get(r, [])
                if not misses_r:
                    continue
                ops = self.rank_planner.plan(
                    rank=r,
                    miss_experts=misses_r,
                    rank_cache=self.cache_view.per_rank[r],
                    demand=phase1.demand,
                    priority=eff_priority,
                )
                all_fetch_ops.extend(ops)

        # routing_map_miss + expert_to_rank
        routing_map_miss: Dict[ExpertKey, int] = {
            op.expert: op.fetcher_rank for op in all_fetch_ops
        }
        routing_map_hit = phase1.hit_launch_info.routing_map_hit

        # Planner invariant #7 (revision.md §1.4): hit / miss MUST be disjoint.
        # Same (layer, expert) being both a hit (resident) and a miss (fetch)
        # would (a) double-route its tokens and (b) corrupt the unified
        # expert_rank_table.  Under correct classify this is impossible; if it
        # ever fires it is a controller bug — fail fast (FATAL) rather than
        # silently overwriting in the dict merge below.
        overlap = set(routing_map_hit) & set(routing_map_miss)
        if overlap:
            raise RuntimeError(
                f"[controller rank={self.ep_rank}] L{layer_id} hit/miss NOT "
                f"disjoint — {sorted(overlap)[:10]} appear as both resident "
                f"hit and planned miss.  classify_and_kick_hit bug.")

        expert_to_rank: Dict[ExpertKey, int] = dict(routing_map_hit)
        expert_to_rank.update(routing_map_miss)

        if _HEAVY:
            n_my_fetch = sum(
                1 for op in all_fetch_ops if op.fetcher_rank == self.ep_rank)
            print(f"[ctrl rank={self.ep_rank}] L{layer_id} P2: "
                  f"total_fetch={len(all_fetch_ops)} my_fetch={n_my_fetch} "
                  f"shadow_after={self.cache_view.total_slots_used()}",
                  flush=True)

        plan = LayerPlan(
            layer_id=layer_id,
            fetch_ops=all_fetch_ops,
            routing_map_miss=routing_map_miss,
            expert_to_rank=expert_to_rank,
            n_unique_demand=phase1.n_unique_demand,
            n_hits=phase1.n_hits,
            n_misses_dedup=phase1.n_misses_dedup,
        )
        if _PLACEMENT_STATS:
            self._accum_placement_stats(plan, phase1, is_decode)
        return plan

    def churn_summary(self) -> dict:
        """expert당 재-fetch(thrash) + rank별 고유 expert 요약."""
        out = {}
        for ph in ("prefill", "decode"):
            ctr = self._churn_ctr[ph]
            if not ctr:
                out[ph] = None; continue
            tot = sum(ctr.values()); dist = len(ctr)
            ge2 = sum(1 for v in ctr.values() if v >= 2)
            ge5 = sum(1 for v in ctr.values() if v >= 5)
            out[ph] = {
                "distinct_fetched": dist, "total_fetches": tot,
                "churn_factor": round(tot / dist, 2),          # expert당 평균 재-fetch
                "frac_refetched_ge2": round(ge2 / dist, 3),    # 한 번이라도 재-fetch된 비율
                "frac_thrash_ge5": round(ge5 / dist, 3),       # 5번+ 재-fetch(심한 thrash)
                "per_rank_distinct": [len(s) for s in self._churn_rankset[ph]],
            }
        return out

    def reset_rank_accum(self) -> None:
        """Start a new eval iteration (new batch of requests): zero per-rank accum."""
        if self._rank_accum is not None:
            self._rank_accum.reset()

    def commit_batch_to_collection(self) -> None:
        """Online history growth: add each rank's accumulated [L,E] (this batch's
        requests) to the retrieval collection.  Identical across ranks -> safe."""
        if self._retrieval is not None and self._rank_accum is not None:
            for g in range(self.ep_size):
                self._retrieval.add(self._rank_accum.rank_matrix(g))

    def reset_placement_stats(self) -> None:
        self.placement_stats = {
            "prefill": _new_placement_bucket(),
            "decode": _new_placement_bucket(),
        }
        self._churn_ctr = {"prefill": {}, "decode": {}}
        self._churn_rankset = {"prefill": [set() for _ in range(self.ep_size)],
                               "decode": [set() for _ in range(self.ep_size)]}

    def _accum_placement_stats(self, plan: LayerPlan, phase1: Phase1Output,
                               is_decode: bool) -> None:
        """Per-layer placement diagnostics (token movement + victim accounting).

        token movement = the all-to-all/NVLink volume implied by routing each
        demanded token to its expert's owner rank: local = stays on the token's
        rank, remote = crosses to another rank.  victim_in_demand = a fetch whose
        evicted resident is itself demanded THIS layer (i.e. evicting an in-use
        expert → archer must STAGE the fetch).
        """
        import numpy as np
        ge = phase1.per_rank_count
        if ge is None:
            return
        ge = np.asarray(ge)  # [G, E] int
        demand = phase1.demand
        b = self.placement_stats["decode" if is_decode else "prefill"]

        local = remote = 0
        for (_l, e), owner in plan.expert_to_rank.items():
            col = ge[:, e]
            loc = int(col[owner]); tot = int(col.sum())
            local += loc; remote += (tot - loc)
        mloc = mrem = 0
        for (_l, e), owner in plan.routing_map_miss.items():
            col = ge[:, e]
            loc = int(col[owner]); tot = int(col.sum())
            mloc += loc; mrem += (tot - loc)

        vid = vemp = vcold = 0
        per_rank_fetch = [0] * self.ep_size
        for op in plan.fetch_ops:
            per_rank_fetch[op.fetcher_rank] += 1
            v = op.victim_expert
            if v is None:
                vemp += 1
            elif v in demand:        # evicting a this-layer hit → staging
                vid += 1
            else:
                vcold += 1
        max_fetch = max(per_rank_fetch) if per_rank_fetch else 0

        demand_tok = local + remote
        miss_tok = mloc + mrem
        b["layers"] += 1
        b["demand_tokens"] += demand_tok
        b["local_tokens"] += local; b["remote_tokens"] += remote
        b["miss_local_tokens"] += mloc; b["miss_remote_tokens"] += mrem
        b["hit_tokens"] += (demand_tok - miss_tok); b["miss_tokens"] += miss_tok
        b["n_fetch"] += len(plan.fetch_ops)
        b["victim_in_demand"] += vid
        b["victim_empty"] += vemp; b["victim_cold"] += vcold
        b["n_hits"] += plan.n_hits; b["n_miss"] += plan.n_misses_dedup
        b["sum_max_fetch"] += max_fetch
        b["sum_mean_fetch"] += len(plan.fetch_ops) / self.ep_size
        # churn: (l,e)별 재-fetch 횟수 + rank별 고유 expert
        ph = "decode" if is_decode else "prefill"
        ctr = self._churn_ctr[ph]; rs = self._churn_rankset[ph]
        for op in plan.fetch_ops:
            ctr[op.expert] = ctr.get(op.expert, 0) + 1
            rs[op.fetcher_rank].add(op.expert)
        # per-rank hit attribution (which rank owns the hit) — dead-rank check
        hbr = b["hits_by_rank"]
        for op in phase1.hit_launch_info.hit_ops:
            hbr[op.owner_rank] = hbr.get(op.owner_rank, 0) + 1
        # occupancy / duplicate snapshot (this layer): slots_used vs distinct.
        # slots_used - unique = redundant copies (capacity wasted on duplicates).
        b["occ_slots_sum"] += self.cache_view.total_slots_used()
        b["occ_unique_sum"] += self.cache_view.global_unique_experts()
        b["occ_samples"] += 1
        opr = b["occ_per_rank_sum"]
        for r, c in enumerate(self.cache_view.per_rank):
            opr[r] = opr.get(r, 0) + (c.cap - c.free_slots())
        bl = b["by_layer"].setdefault(int(plan.layer_id), [0, 0, 0, 0])
        bl[0] += len(plan.fetch_ops); bl[1] += vid
        bl[2] += remote; bl[3] += demand_tok

    # ============================================================
    # Phase 4 — Layer 종료 verify + bump_layer
    # ============================================================
    def verify_layer_end(self, layer_id: int) -> Tuple[int, Set[ExpertKey], Set[ExpertKey]]:
        """shadow == archer_physical assert + 모든 rank bump_layer.

        Returns (drift_count, missing_in_archer, extra_in_archer) for this
        rank's slice.  Under I1-I3 + I5 모두 정상이면 (0, ∅, ∅).
        """
        drift_count = 0
        missing: Set[ExpertKey] = set()
        extra: Set[ExpertKey] = set()
        if _VERIFY_LAYER_END and self._can_query_archer():
            local_dispatcher = self.command_dispatcher.local_dispatcher
            physical = set(local_dispatcher.get_cached_experts(0))
            drift_count, missing, extra = (
                self.cache_view.per_rank[self.ep_rank]
                .verify_against_archer(physical)
            )
            # DEBUG: dump shadow + physical sizes + layer distribution on first
            # drift event so we can see if it's forward#1/L0 vs later layer.
            if drift_count > 0 and self._n_drift_events == 0:
                shadow_keys = self.cache_view.per_rank[self.ep_rank].occupied_keys()
                from collections import Counter as _C
                shadow_layers = _C(k[0] for k in shadow_keys)
                physical_layers = _C(k[0] for k in physical)
                print(f"[ctrl rank={self.ep_rank}] L{layer_id} FIRST DRIFT "
                      f"shadow_count={len(shadow_keys)} "
                      f"physical_count={len(physical)} "
                      f"shadow_layers={dict(shadow_layers)} "
                      f"physical_layers={dict(physical_layers)}",
                      flush=True)
            if drift_count > 0:
                self._n_drift_events += 1
                self._cumulative_drift += drift_count
                if self._n_drift_events <= self._n_unconditional_drift_logs:
                    print(f"[ctrl rank={self.ep_rank}] L{layer_id} DRIFT "
                          f"count={drift_count} missing_in_archer={sorted(missing)[:10]} "
                          f"extra_in_archer={sorted(extra)[:10]}", flush=True)
                # I1 위반 — 새 design 에선 정상 동작 중 불가능.  raise.
                raise RuntimeError(
                    f"[controller rank={self.ep_rank}] L{layer_id} shadow/"
                    f"archer drift={drift_count}.  This is a correctness bug "
                    f"under the cooperative-offloading invariants.  "
                    f"missing_in_archer={sorted(missing)} "
                    f"extra_in_archer={sorted(extra)}")
        # 모든 rank bump_layer
        self.cache_view.bump_all_layers()
        return drift_count, missing, extra

    # ============================================================
    # internals
    # ============================================================
    def _can_query_archer(self) -> bool:
        cd = self.command_dispatcher
        if cd is None:
            return False
        ld = getattr(cd, "local_dispatcher", None)
        return ld is not None and hasattr(ld, "get_cached_experts")

    def _stub_per_rank_count(self, local_router_mask: torch.Tensor) -> torch.Tensor:
        """ep_size=1 / test 경로용 stub.  실제 NCCL all_gather 없이 자기 mask 만."""
        local_count = (
            local_router_mask.view(-1, self.num_experts)
            .sum(dim=0).to(torch.int64).cpu()
        )
        per_rank = torch.zeros(
            self.ep_size, self.num_experts, dtype=torch.int64)
        per_rank[self.ep_rank] = local_count
        return per_rank
