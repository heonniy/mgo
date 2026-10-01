"""RankPlanner — Phase 2 의 per-rank sequential plan.

Design 참조: design.md §2 Phase 2

핵심 알고리즘 (각 rank 가 자기가 책임진 miss set 에 대해):
    1. demand 내림차순 정렬       → fetch 우선순위
    2. sequential 루프:
       a. empty slot 우선
       b. 없으면 evict_policy.pick_victim
       c. FetchOp 생성 (dst_slot, victim 박힘)
       d. rank_cache.apply(...)  ← ★ 다음 iter 의 pick_victim 이 새로 갱신된
                                    state 를 보도록 즉시 mutate

sequential + 즉시 apply 가 두 가지 정상 동작 invariant 를 자연 보장:
    (1) 서로 다른 두 expert 가 같은 empty slot 을 노리는 중복 victim 불가능
        (apply 직후 그 slot 은 occupied 라 다음 iter 가 다른 slot 또는 victim 선택)
    (2) 방금 1순위로 넣은 expert 를 2순위의 victim 으로 즉시 쫓아내는 fetch
        낭비 불가능 (apply 직후 그 expert 의 meta.last_used 가 현재 layer_seq
        라 evict_policy 가 자연히 다른 victim 선택)

I3: shadow 가 Archer 대기 없이 즉시 "layer 종료 후 상태" 로 갱신 완료.

ASSUMPTION (입력):
    miss_experts 는 union dedup 후 owner_policy 가 이 rank 에 배정한 것.
    즉 다른 rank 가 같은 expert 를 fetch 하지 않음.  중복 worry 없음.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .evict_policy import EvictionPolicy
from .layer_plan import FetchOp
from .slot_cache import ExpertKey, RankSlotCache


class RankPlanner:
    """Phase 2 의 sequential planner."""

    def __init__(self, evict_policy: EvictionPolicy):
        self.evict_policy = evict_policy

    def plan(
        self,
        rank: int,
        miss_experts: List[ExpertKey],
        rank_cache: RankSlotCache,
        demand: Dict[ExpertKey, int],
        priority=None,
    ) -> List[FetchOp]:
        """Per-rank sequential plan.  shadow 를 mutate.  fetch_ops 반환.

        miss_experts: 이 rank 가 책임지는 miss set (owner_policy 출력).
        rank_cache:   이 rank 의 RankSlotCache (mutate 됨).
        demand:       global expert → token count.
        priority:     optional EAMC [num_layers, num_experts] matrix; EAMC
                      evict_policy 만 사용 (나머지 무시).

        Returns: fully-resolved fetch_ops (dst_slot, victim_expert 박힘).
        """
        # Step 5: demand 내림차순.  tie-break by (layer, expert) 로 결정성.
        ordered = sorted(
            miss_experts,
            key=lambda k: (-int(demand.get(k, 0)), k[0], k[1]),
        )

        fetch_ops: List[FetchOp] = []
        order = 0  # rank-local FIFO order = PlanQueue 삽입 순서 = 실제 H2D 순서
        for expert in ordered:
            # 이미 resident 이면 skip (이론적으로 owner_policy 가 dedup 후라
            # 안 와야 하지만 방어).  cache_view 가 mutate 되었으므로 매번 체크.
            if rank_cache.is_resident(expert):
                continue

            # Step 6: empty slot 우선, 없으면 pick_victim
            empty = rank_cache.first_empty_slot()
            if empty is not None:
                dst_slot = empty
                victim: Optional[ExpertKey] = None
            else:
                victim_slot = self.evict_policy.pick_victim(
                    rank_cache, incoming=expert, demand=demand,
                    priority=priority)
                if victim_slot is None:
                    # 정책이 None 반환 — cap=0 또는 모든 slot 이 protected (구
                    # architecture 잔재).  새 design 은 protected 없으니 발생
                    # 안 함.  defensive: drop 처리.
                    continue
                dst_slot = victim_slot
                victim = rank_cache.slots[dst_slot]

            # Step 7: FetchOp emit (fully-resolved, rank-local order 박힘)
            fetch_ops.append(FetchOp(
                expert=expert,
                fetcher_rank=rank,
                dst_slot=dst_slot,
                victim_expert=victim,
                order=order,
                source="pcie",
            ))
            order += 1

            # ★ apply 즉시 — 다음 iter 의 pick_victim 이 새 state 를 봄
            rank_cache.apply(
                slot=dst_slot,
                evict=victim,
                insert=expert,
                demand_count=int(demand.get(expert, 0)),
            )

        return fetch_ops
