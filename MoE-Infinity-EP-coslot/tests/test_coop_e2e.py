"""E2E tests for the cooperative-offloading controller flow.

NCCL / archer 없는 환경에서 controller 의 Phase 1→2→4 결정성과 invariant
들을 검증.  모든 rank 에 별도 GlobalCacheController 인스턴스를 만들고,
같은 router_mask 시퀀스를 입력으로 줬을 때:

  T1. 결정성  — 모든 rank 의 fetch_ops, expert_to_rank, shadow 가 byte-identical
  T2. union dedup — 같은 expert 가 두 rank 의 demand 에 있으면 fetch 가 1회만
  T3. cap 압박 시 LRU evict 정확성 — 방금 install 한 expert 가 즉시 victim 안 됨
  T4. (layer, expert) key invariant — 다른 layer 의 같은 expert_id 는 별개
  T5. Phase 4 verify — shadow == FakeArcher 의 cached_set

FakeArcher 는 controller 가 issue 한 explicit_evict / explicit_fetch_async
명령을 set 단위로 미러링 (cudaMemcpy 없음, 단순 bookkeeping).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import pytest
import torch

from moe_infinity_ep.controller.global_controller import GlobalCacheController
from moe_infinity_ep.controller.owner_policy import (
    StaticPlacement, BalancedPlacement,
)
from moe_infinity_ep.controller.evict_policy import LRUEviction, LFUEviction
from moe_infinity_ep.controller.rank_planner import RankPlanner
from moe_infinity_ep.controller.command_dispatcher import CommandDispatcher
from moe_infinity_ep.controller.layer_plan import ExpertKey


# ============================================================
# Mocks
# ============================================================
class FakeArcher:
    """coslot archer ExpertDispatcher mock.  ``submit_plan`` mirrors the C++
    commit bookkeeping (apply miss_ops in order: evict victim, install expert)
    so ``get_cached_experts`` ends byte-identical with the controller shadow.
    """

    def __init__(self):
        self.cached: Set[ExpertKey] = set()
        self.calls: List[Tuple[str, ExpertKey]] = []

    def submit_plan(self, gpu, hit_ops, miss_ops):
        # hits stay resident (no change).  Apply misses in rank-local order.
        for (l, e, dst, vl, ve, order) in sorted(miss_ops, key=lambda t: t[5]):
            if vl >= 0:
                self.cached.discard((vl, ve))
                self.calls.append(("evict", (vl, ve)))
            self.cached.add((l, e))
            self.calls.append(("fetch", (l, e)))

    def wait_layer_done(self):
        return None

    def get_cached_experts(self, gpu_id):
        # archer 의 실제 API 는 tuple list 반환
        return [(l, e) for (l, e) in self.cached]


class CollectiveMock:
    """ep_size 개 controller 가 공유하는 'cross-rank' shared buffer.

    실제 NCCL all_gather 의 결과를 단일 process 안에서 simulate.
    매 layer 마다 모든 rank 의 router_mask 를 한 곳에 모으고, 그 count
    합을 demand_collector.collect_with_count 가 반환할 per_rank tensor 로
    구성.
    """

    def __init__(self, ep_size: int, num_experts: int):
        self.ep_size = ep_size
        self.num_experts = num_experts
        # rank 별 이번 layer 의 router_mask 가 채워질 곳
        self._cur_masks: Dict[int, torch.Tensor] = {}
        self._per_rank_cache: Optional[torch.Tensor] = None

    def submit_rank_mask(self, rank: int, router_mask: torch.Tensor) -> None:
        self._cur_masks[rank] = router_mask
        self._per_rank_cache = None  # invalidate

    def all_gather_count(self) -> torch.Tensor:
        """per_rank[ep_size, num_experts] int64."""
        if self._per_rank_cache is None:
            assert len(self._cur_masks) == self.ep_size, (
                f"need {self.ep_size} ranks submitted, got "
                f"{sorted(self._cur_masks.keys())}")
            counts = []
            for r in range(self.ep_size):
                m = self._cur_masks[r]
                counts.append(
                    m.view(-1, self.num_experts).sum(dim=0).to(torch.int64))
            self._per_rank_cache = torch.stack(counts, dim=0).cpu()
        return self._per_rank_cache

    def end_layer(self):
        self._cur_masks.clear()
        self._per_rank_cache = None


class MockDemandCollector:
    """controller.demand_collector slot 에 들어가는 mock.  CollectiveMock 의
    all_gather_count 결과를 그대로 반환."""

    def __init__(self, collective: CollectiveMock):
        self.collective = collective
        # signature 호환 — 실제 DemandCollector 와 동일 attr
        self.ep_size = collective.ep_size
        self.num_experts = collective.num_experts
        self.ep_group = None

    def collect_with_count(self, layer_id, local_router_mask):
        return self.collective.all_gather_count()


# ============================================================
# Helpers
# ============================================================
def _make_controllers(
    ep_size: int, cap: int, num_layers: int, num_experts: int,
    owner_cls=StaticPlacement, evict_cls=LRUEviction,
):
    """ep_size 개의 controller + FakeArcher 인스턴스 생성, wiring 완료."""
    collective = CollectiveMock(ep_size, num_experts)
    controllers = []
    archers = []
    for r in range(ep_size):
        archer = FakeArcher()
        c = GlobalCacheController(
            ep_size=ep_size, ep_rank=r, cap_per_rank=cap,
            num_layers=num_layers, num_experts=num_experts,
            ep_group=None,
            owner_policy=owner_cls(),
            rank_planner=RankPlanner(evict_cls()),
            demand_collector=MockDemandCollector(collective),
        )
        c.command_dispatcher = CommandDispatcher(
            archer_engine=None, local_dispatcher=archer, rank=r)
        controllers.append(c)
        archers.append(archer)
    return controllers, archers, collective


def _run_one_layer(
    controllers, archers, collective,
    layer_id: int, per_rank_router_masks: List[torch.Tensor],
):
    """모든 rank 에서 동시 (1 forward 의 1 layer) 진행."""
    ep_size = len(controllers)
    # 각 rank 가 자기 router_mask 를 collective 에 제출 (NCCL all_gather sim)
    for r in range(ep_size):
        collective.submit_rank_mask(r, per_rank_router_masks[r])

    p1_list = []
    for r in range(ep_size):
        controllers[r].begin_layer(layer_id)
        p1 = controllers[r].classify_and_kick_hit(
            layer_id, per_rank_router_masks[r])
        p1_list.append(p1)

    plan_list = []
    for r in range(ep_size):
        plan = controllers[r].plan_misses(p1_list[r], layer_id)
        plan_list.append(plan)

    # archer 명령 issue (coslot: single submit_plan per rank)
    for r in range(ep_size):
        controllers[r].command_dispatcher.submit_plan(
            plan_list[r], p1_list[r].hit_launch_info.hit_ops, my_rank=r)

    # Phase 4 verify
    for r in range(ep_size):
        controllers[r].verify_layer_end(layer_id)

    collective.end_layer()
    return p1_list, plan_list


# ============================================================
# T1. determinism — 모든 rank 의 결정이 byte-identical
# ============================================================
def test_T1_determinism_across_ranks_first_layer():
    """ep_size=4 cap=∞ (=num_layers×num_experts).  첫 layer 의 plan 이 모든
    rank 에서 동일한 expert_to_rank, fetch_ops, shadow 를 produce."""
    ep_size = 4
    num_experts = 16
    num_layers = 4
    cap = 64  # 큰 cap
    controllers, archers, coll = _make_controllers(
        ep_size, cap, num_layers, num_experts)

    # rank 0 의 token: expert {0, 3, 5, 7}
    # rank 1: {1, 3, 5, 9}
    # rank 2: {0, 4, 8}
    # rank 3: {2, 6, 10, 12}
    def mk(experts):
        m = torch.zeros(4, num_experts, dtype=torch.bool)
        for i, e in enumerate(experts):
            m[i % 4, e] = True
        return m
    masks = [
        mk([0, 3, 5, 7]),
        mk([1, 3, 5, 9]),
        mk([0, 4, 8]),
        mk([2, 6, 10, 12]),
    ]

    p1s, plans = _run_one_layer(controllers, archers, coll, 0, masks)

    # 모든 rank 의 plan.expert_to_rank 가 동일
    base = plans[0].expert_to_rank
    for r in range(1, ep_size):
        assert plans[r].expert_to_rank == base, (
            f"rank {r} expert_to_rank diverged: "
            f"{plans[r].expert_to_rank} vs {base}")

    # 모든 rank 의 fetch_ops 가 동일 set
    base_set = {(op.expert, op.fetcher_rank, op.dst_slot, op.victim_expert)
                for op in plans[0].fetch_ops}
    for r in range(1, ep_size):
        s = {(op.expert, op.fetcher_rank, op.dst_slot, op.victim_expert)
             for op in plans[r].fetch_ops}
        assert s == base_set, f"rank {r} fetch_ops diverged"

    # union dedup 검증: union = {0,1,2,3,4,5,6,7,8,9,10,12} = 12 unique
    assert p1s[0].n_unique_demand == 12
    # 같은 expert 가 다른 rank 의 demand 에도 있어도 fetch 는 1개
    expert_ids_in_fetches = [op.expert[1] for op in plans[0].fetch_ops]
    assert len(expert_ids_in_fetches) == len(set(expert_ids_in_fetches)) == 12


# ============================================================
# T2. union dedup — 같은 expert 가 두 rank demand 시 fetch 1회만
# ============================================================
def test_T2_union_dedup():
    ep_size = 4
    num_experts = 8
    cap = 32
    controllers, archers, coll = _make_controllers(ep_size, cap, 1, num_experts)

    # 4 rank 가 모두 expert 0 demand (token 수만 다름)
    masks = []
    for r in range(ep_size):
        m = torch.zeros(r + 1, num_experts, dtype=torch.bool)
        m[:, 0] = True  # 모두 expert 0 만
        masks.append(m)

    p1s, plans = _run_one_layer(controllers, archers, coll, 0, masks)

    # union demand = 1 (expert 0 만)
    assert p1s[0].n_unique_demand == 1
    # miss_dedup = 1
    assert p1s[0].n_misses_dedup == 1
    # fetch_ops 도 1 (4 rank 가 동시에 fetch 하지 않음)
    assert len(plans[0].fetch_ops) == 1
    # demand count 가 합산됨 (4 rank 의 sum)
    expected_demand = 1 + 2 + 3 + 4
    actual_demand = p1s[0].demand[(0, 0)]
    assert actual_demand == expected_demand, (
        f"demand for (0,0): expected {expected_demand}, got {actual_demand}")


# ============================================================
# T3. cap 압박 — 방금 install 한 expert 가 즉시 victim 안 됨
# ============================================================
def test_T3_sequential_apply_invariant():
    """sequential apply invariant: cap 이 이번 layer 의 demand 를 수용할 수
    있을 때, 같은 layer 안에서 방금 install 한 expert 가 동일 layer 의 다음
    install 의 victim 으로 picking 되지 않음.

    핵심: ``apply`` 가 매 호출 ``touch_counter += 1`` → 새로 install 한 expert
    의 ``last_used`` 가 가장 최신 → LRU 가 자연히 head 에서 밀어냄.

    시나리오: cap=4 에 직전 layer 의 4 expert 가 들어 있음.  새 layer 의
    install 3개 → 3 개 victim 필요.  victim 들은 모두 직전 layer 의 expert
    여야 함 (이번 layer 의 install 끼리 victim 관계 없음).
    """
    ep_size = 1
    num_experts = 16
    cap = 4
    controllers, archers, coll = _make_controllers(
        ep_size, cap, 2, num_experts)

    # Layer 0: cache 채우기 — 4 expert install
    m0 = torch.zeros(1, num_experts, dtype=torch.bool)
    for e in [0, 1, 2, 3]:
        m0[0, e] = True
    _run_one_layer(controllers, archers, coll, 0, [m0])
    rc = controllers[0].cache_view.per_rank[0]
    assert rc.occupied_keys() == {(0, 0), (0, 1), (0, 2), (0, 3)}

    # Layer 1: 새 expert {10, 11, 12} demand → 3 install, 3 victim
    m1 = torch.zeros(1, num_experts, dtype=torch.bool)
    for e in [10, 11, 12]:
        m1[0, e] = True
    p1, plans = _run_one_layer(controllers, archers, coll, 1, [m1])

    plan = plans[0]
    assert len(plan.fetch_ops) == 3
    # 모든 install 이 victim 필요 (cap 가득)
    for op in plan.fetch_ops:
        assert op.victim_expert is not None

    # 핵심 invariant: 같은 layer 의 install 끼리는 victim 관계 없음.
    install_set = {op.expert for op in plan.fetch_ops}
    victim_set = {op.victim_expert for op in plan.fetch_ops
                  if op.victim_expert is not None}
    common = install_set & victim_set
    assert not common, (
        f"sequential apply invariant violated — installed expert(s) {common} "
        f"got picked as victim within same layer")

    # victim 들은 모두 직전 layer 의 expert
    expected_victims = {(0, 0), (0, 1), (0, 2), (0, 3)}
    assert victim_set <= expected_victims, (
        f"victims {victim_set} should be subset of prior-layer experts "
        f"{expected_victims}")

    # 최종 shadow: cap=4 유지
    assert rc.cap - rc.free_slots() == 4
    # Layer 1 의 3 expert + 직전 layer 의 1 expert 남음
    occ = rc.occupied_keys()
    new_count = sum(1 for (l, _) in occ if l == 1)
    old_count = sum(1 for (l, _) in occ if l == 0)
    assert new_count == 3 and old_count == 1


# ============================================================
# T4. (layer, expert) key invariant — 같은 expert_id 다른 layer = 별개
# ============================================================
def test_T4_layer_expert_key_invariant():
    ep_size = 1
    num_experts = 4
    cap = 8
    controllers, archers, coll = _make_controllers(ep_size, cap, 3, num_experts)

    def mk(expert_list):
        m = torch.zeros(1, num_experts, dtype=torch.bool)
        for e in expert_list:
            m[0, e] = True
        return m

    # Layer 0: expert {0, 1, 2}
    p10, plans0 = _run_one_layer(controllers, archers, coll, 0, [mk([0, 1, 2])])
    # 모두 miss → 3 install
    assert len(plans0[0].fetch_ops) == 3

    # Layer 1: 같은 expert_id 들 {0, 1, 2} 다시 demand
    p11, plans1 = _run_one_layer(controllers, archers, coll, 1, [mk([0, 1, 2])])
    # Layer 1 의 (1, 0/1/2) 는 cache 에 없음 → 모두 miss
    assert p11[0].n_hits == 0
    assert p11[0].n_misses_dedup == 3
    assert len(plans1[0].fetch_ops) == 3

    # Layer 1 다시 (decode token 1) — 이번엔 같은 layer 의 같은 expert → hit
    p11b, plans1b = _run_one_layer(controllers, archers, coll, 1, [mk([0, 1])])
    assert p11b[0].n_hits == 2 and p11b[0].n_misses_dedup == 0


# ============================================================
# T5. Phase 4 verify — shadow == FakeArcher.cached
# ============================================================
def test_T5_shadow_matches_archer():
    ep_size = 4
    num_experts = 16
    cap = 8
    controllers, archers, coll = _make_controllers(
        ep_size, cap, 4, num_experts)

    def mk(experts, n_tokens=2):
        m = torch.zeros(n_tokens, num_experts, dtype=torch.bool)
        for i, e in enumerate(experts):
            m[i % n_tokens, e] = True
        return m

    # 4 layer, 다양한 demand
    sequences = [
        ([0, 3, 5], [1, 5, 9], [2, 4, 8], [6, 10, 11]),     # layer 0
        ([1, 3], [5, 7], [2, 4, 12], [0, 6, 8]),            # layer 1
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10], [11, 12, 13]),  # layer 2
        ([0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13]),   # layer 3
    ]
    for layer_id, seqs in enumerate(sequences):
        masks = [mk(list(s)) for s in seqs]
        _run_one_layer(controllers, archers, coll, layer_id, masks)

    # 모든 rank 에서 shadow == archer
    for r in range(ep_size):
        shadow = controllers[r].cache_view.per_rank[r].occupied_keys()
        physical = set(archers[r].cached)
        assert shadow == physical, (
            f"rank {r} drift: shadow {sorted(shadow)} != "
            f"archer {sorted(physical)}")


# ============================================================
# T6. StaticPlacement vs BalancedPlacement — placement 정책 차이 검증
# ============================================================
@pytest.mark.parametrize(
    "owner_cls", [StaticPlacement, BalancedPlacement])
def test_T6_owner_policy_drives_fetch_distribution(owner_cls):
    ep_size = 4
    num_experts = 32
    cap = 16
    controllers, archers, coll = _make_controllers(
        ep_size, cap, 2, num_experts, owner_cls=owner_cls)

    def mk(experts):
        m = torch.zeros(1, num_experts, dtype=torch.bool)
        for e in experts:
            m[0, e] = True
        return m
    # 모든 rank 가 expert {0..15} 의 일부 demand → union 16
    masks = [mk(list(range(0, 16, 1))), mk(list(range(0, 16, 1))),
             mk(list(range(0, 16, 1))), mk(list(range(0, 16, 1)))]
    p1s, plans = _run_one_layer(controllers, archers, coll, 0, masks)

    counts = [0] * ep_size
    for op in plans[0].fetch_ops:
        counts[op.fetcher_rank] += 1

    if owner_cls is StaticPlacement:
        # expert_id % 4 → [4, 4, 4, 4]
        assert counts == [4, 4, 4, 4]
    else:  # BalancedPlacement
        # 16 misses 를 4 rank 에 균등 → [4, 4, 4, 4]
        assert max(counts) - min(counts) <= 1


# ============================================================
# T7. LFU evict policy — freq 낮은 expert 가 먼저 victim
# ============================================================
def test_T7_lfu_eviction():
    ep_size = 1
    num_experts = 8
    cap = 3
    controllers, archers, coll = _make_controllers(
        ep_size, cap, 5, num_experts, evict_cls=LFUEviction)

    def mk(experts):
        m = torch.zeros(1, num_experts, dtype=torch.bool)
        for e in experts:
            m[0, e] = True
        return m

    # Layer 0: cache 비어있음.  expert {0,1,2} 모두 miss → install at cap=3.
    #   freq: (0,0)=1, (0,1)=1, (0,2)=1
    _run_one_layer(controllers, archers, coll, 0, [mk([0, 1, 2])])
    rc = controllers[0].cache_view.per_rank[0]
    assert {(0, 0), (0, 1), (0, 2)} == rc.occupied_keys()
    # Layer 1: (1, 1) miss → install.  cap 가득 (3), LFU victim 선택.
    #   freq 동률 (모두 1) 이라 tie-break by last_used → (0,0) 이 가장 오래된
    #   → (0,0) victim.  install (1, 1) at slot 0.
    _run_one_layer(controllers, archers, coll, 1, [mk([1])])
    assert {(0, 1), (0, 2), (1, 1)} == rc.occupied_keys()
    # Layer 0 다시: hit (0,1), (0,2).  freq+=1 each.
    #   freq: (0,1)=2, (0,2)=2, (1,1)=1
    _run_one_layer(controllers, archers, coll, 0, [mk([1, 2])])
    # Layer 0 의 demand {1, 2, 7}: hit (0,1)(0,2), miss (0,7).
    #   cap 가득 → LFU victim = freq 최소 = (1,1).
    p1, plans = _run_one_layer(
        controllers, archers, coll, 0, [mk([1, 2, 7])])
    assert len(plans[0].fetch_ops) == 1
    assert plans[0].fetch_ops[0].expert == (0, 7)
    assert plans[0].fetch_ops[0].victim_expert == (1, 1), (
        f"LFU should pick (1,1) (freq=1) but got "
        f"{plans[0].fetch_ops[0].victim_expert}")


# ============================================================
# T8. Phase 4 drift — explicit_evict 외부에서 미스매치 만들면 RuntimeError
# ============================================================
def test_T8_phase4_drift_raises():
    import os
    os.environ["MOE_EP_VERIFY_LAYER_END"] = "1"
    ep_size = 1
    num_experts = 4
    cap = 4
    controllers, archers, coll = _make_controllers(
        ep_size, cap, 1, num_experts)

    m = torch.zeros(1, num_experts, dtype=torch.bool)
    m[0, 0] = True
    m[0, 1] = True
    _run_one_layer(controllers, archers, coll, 0, [m])
    # 정상 → drift=0

    # 외부 oracle 로 archer 만 mutate (controller bypass) → drift
    archers[0].cached.add((0, 99))  # archer 에만 가짜 entry

    # 다음 layer 호출 시 Phase 4 verify 에서 drift detect → raise
    m2 = torch.zeros(1, num_experts, dtype=torch.bool)
    m2[0, 2] = True
    with pytest.raises(RuntimeError, match=r"drift"):
        _run_one_layer(controllers, archers, coll, 0, [m2])


# ============================================================
# T-hit-victim. Round-7 regression guard — a HIT expert demanded in the
# current layer pass must NOT be selected as a victim for a same-pass miss.
#
# Round-7 bug: while processing layer L, a miss replace reused the slot of a
# still-needed HIT expert (same (layer,expert) resident + re-demanded), which
# on the GPU corrupted the in-flight hit GEMM and on the controller drifted the
# bookkeeping.  Fix: Phase-1.3 touches hits (most-recent) so LRU never evicts a
# demanded hit; the executor (Option A) computes hits before issuing miss
# fetches.  This test pins the controller-side half: demanded hits stay
# resident, victims come only from non-demanded residents, drift=0.
# ============================================================
def test_hit_victim_drift_free():
    ep_size = 1
    num_experts = 32
    cap = 4
    L = 5
    controllers, archers, coll = _make_controllers(
        ep_size, cap, L + 1, num_experts)

    def mk(experts):
        m = torch.zeros(1, num_experts, dtype=torch.bool)
        for e in experts:
            m[0, e] = True
        return m

    # Pass 1 of layer L: install {0,1,2,3} → cache full at cap=4.
    _run_one_layer(controllers, archers, coll, L, [mk([0, 1, 2, 3])])
    rc = controllers[0].cache_view.per_rank[0]
    assert rc.occupied_keys() == {(L, 0), (L, 1), (L, 2), (L, 3)}

    # Pass 2 of layer L (e.g. next decode token): demand {0, 1, 20}.
    #   (L,0),(L,1) are HITS (resident + demanded).  (L,20) is a miss needing
    #   1 slot → 1 eviction from the full cache.
    p1s, plans = _run_one_layer(controllers, archers, coll, L, [mk([0, 1, 20])])
    p1 = p1s[0]
    plan = plans[0]

    assert p1.n_hits == 2, f"expected 2 hits (0,1), got {p1.n_hits}"
    assert p1.n_misses_dedup == 1, f"expected 1 miss (20), got {p1.n_misses_dedup}"
    assert len(plan.fetch_ops) == 1
    op = plan.fetch_ops[0]
    assert op.expert == (L, 20)
    # The victim must be a NON-demanded resident ({2,3}), never a demanded hit.
    assert op.victim_expert in {(L, 2), (L, 3)}, (
        f"hit-victim regression: victim {op.victim_expert} is a demanded hit "
        f"or unexpected key (allowed: (L,2)/(L,3))")
    # Demanded hits survive; the new miss is resident; drift=0.
    occ = rc.occupied_keys()
    assert (L, 0) in occ and (L, 1) in occ, "demanded hit was wrongly evicted"
    assert (L, 20) in occ, "promoted miss not resident"
    assert occ == set(archers[0].cached), (
        f"drift: shadow {sorted(occ)} != archer {sorted(archers[0].cached)}")
