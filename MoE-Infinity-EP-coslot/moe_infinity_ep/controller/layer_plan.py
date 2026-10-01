"""LayerPlan + 관련 dataclass — Cooperative offloading 새 architecture.

Phase 1 + Phase 2 의 산출물을 담는 typed container. 각 dataclass 의 의미와
어떤 phase 에서 채워지는지가 명시적이라 dispatcher / executor 에서
"이건 hit early-kick 입력, 저건 miss plan 입력" 헷갈리지 않음.

Design 참조:
    /home/work/hyewon.lee/실험/main_exp/moe_cooperative_offloading_design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


ExpertKey = Tuple[int, int]  # (layer_id, expert_id)


@dataclass(frozen=True)
class HitOp:
    """이번 layer 에서 cache 에 이미 있는 expert. weight fetch 불필요.

    owner_rank 와 slot 은 Phase 0 의 cache_view 에서 즉시 알 수 있음.
    slot 은 Phase 3 의 hit GEMM 완료 event 추적 (archer 의 Node.compute_event).
    """
    expert: ExpertKey
    owner_rank: int
    slot: int


@dataclass(frozen=True)
class FetchOp:
    """이번 layer 에서 fetch 가 필요한 miss expert.

    fully-resolved: dst_slot, victim_expert 까지 박혀 있음.  archer 측에서는
    이 단위로 ``ExplicitReplaceAsync(victim, new_expert, dst_slot)`` 한 번에
    처리 (S6).
    """
    expert: ExpertKey
    fetcher_rank: int
    dst_slot: int
    victim_expert: Optional[ExpertKey]  # None = empty slot 이었음
    order: int = 0                      # rank-local FIFO order (PlanQueue 순서)
    source: str = "pcie"                # "pcie" or "nvlink_from_rank_N"


@dataclass
class HitLaunchInfo:
    """Phase 1.3 의 hit early-kick 출력.

    이걸 받자마자 ep_executor 가 NCCL hit a2a 를 비동기 launch (background
    stream). controller 는 그 동안 Phase 2 planning 진행.
    """
    layer_id: int
    hit_ops: List[HitOp] = field(default_factory=list)
    # routing_map_hit[expert] = owner_rank.  pack_tokens 의 input.
    routing_map_hit: Dict[ExpertKey, int] = field(default_factory=dict)


@dataclass
class LayerPlan:
    """Phase 2 끝의 통합 plan.  ep_executor 가 Phase 3 실행 입력으로 사용.

    fetch_ops 는 fully-resolved.  routing_map_miss 가 hit 와 분리되어 있어
    miss a2a 를 hit a2a 와 별개 NCCL collective 로 launch 가능 (결정 A).

    expert_to_rank 는 hit+miss 통합 routing 으로, pack_tokens 가 단일
    expert_rank_table 로 합쳐서 token routing 결정 시 사용 가능 (선택).
    그러나 결정 A 에 따라 hit_routing 과 miss_routing 을 따로 보내는 게
    기본이라 expert_to_rank 는 verify / debug 용으로 보존.
    """
    layer_id: int
    fetch_ops: List[FetchOp] = field(default_factory=list)
    # routing_map_miss[expert] = fetcher_rank (그 rank 가 책임지고 fetch + execute)
    routing_map_miss: Dict[ExpertKey, int] = field(default_factory=dict)
    # 통합 routing (Phase 1 + Phase 2 결과 머지) — debug + verify
    expert_to_rank: Dict[ExpertKey, int] = field(default_factory=dict)
    # 통계 / drift instrumentation
    n_unique_demand: int = 0
    n_hits: int = 0
    n_misses_dedup: int = 0


@dataclass
class Phase1Output:
    """Phase 1 끝의 산출물 묶음.  ep_executor 가 Phase 1.3 의 hit_launch_info
    를 즉시 사용하고, miss_per_rank + global demand 를 Phase 2 입력으로 넘김.
    """
    hit_launch_info: HitLaunchInfo
    miss_per_rank: Dict[int, List[ExpertKey]]
    demand: Dict[ExpertKey, int]
    # Union-deduped miss set for this layer (sorted by (layer, expert)).  Always
    # populated.  Baselines route misses via ``miss_per_rank`` (owner_policy);
    # OURS joint placement reads this raw set in ``plan_misses`` instead (its
    # ownership is decided jointly with slot/order, so ``miss_per_rank`` is left
    # empty in ours mode).
    miss_set: List[ExpertKey] = field(default_factory=list)
    # 통계
    n_unique_demand: int = 0
    n_hits: int = 0
    n_misses_dedup: int = 0
    # 2026-05-28 fast-a2a: Phase-1 all_gather 의 per-rank per-expert demand
    # COUNT 행렬을 그대로 보관.  ep_executor 가 routing_map 과 결합해
    # send/recv split counts 를 collective 없이 유도 (nvlink_router.
    # derive_split_counts).  shape [ep_size, num_experts] int64, CPU.
    per_rank_count: object = None
