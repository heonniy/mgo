"""EvictionPolicy — rank-local victim 선정.

Design 참조: design.md §4

``pick_victim`` 은 ``rank_cache`` 를 **읽기만** 한다.  mutate 는 caller
(``RankPlanner``) 의 ``apply`` 가 담당.  sequential 전제: 매 fetch 마다 호출,
직전 ``apply`` 까지 반영된 현재 상태를 본다.

4 정책:
  * LRUEviction         : ``meta.last_used`` 최소
  * LFUEviction         : ``meta.freq`` 최소, tie-break by last_used → slot
  * DemandAwareEviction : demand[expert] 최소인 expert 가 victim
                          (이번 layer 에서 가장 token 적은 expert)
  * PriorityEviction    : EAMC priority 최소인 expert 가 victim.  priority 는
                          controller 가 plan 전에 주입하는 [num_layers,
                          num_experts] matrix (priority_aggregator 산출).
                          이번 layer demand 에 있는 expert 는 보호(원본
                          protected_ondemand 의미) — 곧 compute 할 것이라
                          내리지 않음.  Archer 로의 EAMC eviction 이전의
                          controller-side victim 선택부.

새 정책 추가:
  ``pick_victim(rank_cache, incoming, demand, priority=None) -> SlotId`` 만
  override.  rank_cache 의 empty slot 처리는 RankPlanner 가 함수 호출 전에 함.
  따라서 이 함수가 호출될 때는 항상 cache 가 full (occupied slots 만).
  ``priority`` 는 EAMC 정책만 사용; 나머지는 무시(서명 호환용).
"""
from __future__ import annotations

import random
from typing import Dict, Optional, Protocol

from .slot_cache import ExpertKey, RankSlotCache


def _stable_seed(*vals: int) -> int:
    """Process-independent deterministic seed (see owner_policy._stable_seed).

    NOT builtin hash (salted per-process) — every rank must seed identically so
    the replicated shadow stays byte-identical.
    """
    s = 0
    for v in vals:
        s = (s * 1000003 + int(v)) & 0xFFFFFFFF
    return s


class EvictionPolicy(Protocol):
    """rank-local victim 선정 protocol."""
    name: str

    def pick_victim(
        self,
        rank_cache: RankSlotCache,
        incoming: ExpertKey,
        demand: Dict[ExpertKey, int],
        priority=None,
    ) -> Optional[int]:
        """Return slot index to evict, or None if no eviction possible.

        Pre-condition: rank_cache.free_slots() == 0 (caller ensures).
        Returns: int slot in [0, rank_cache.cap), or None.
        ``priority`` is an optional [num_layers, num_experts] matrix used only
        by EAMC; other policies ignore it.
        """
        ...


class LRUEviction:
    """meta.last_used 최소.  tie-break by inserted_at → slot index."""

    name = "lru"

    def pick_victim(self, rank_cache, incoming, demand, priority=None):
        best_slot = None
        best_score = None
        for slot, meta in rank_cache.occupied_slots():
            score = (meta.last_used, meta.inserted_at, slot)
            if best_score is None or score < best_score:
                best_score = score
                best_slot = slot
        return best_slot


class LFUEviction:
    """meta.freq 최소.  tie-break by last_used → inserted_at → slot.

    freq = ACCESS COUNT (호출 횟수, +1 per layer-use, 토큰 수 무관) — classic LFU,
    MoE-Infinity ``entry.visit``와 동일 정의.  tie-break은 deterministic 위해
    LRU(last_used)→inserted_at→slot (MoE-Infinity는 tie-break 없음=삽입순).
    """

    name = "lfu"

    def pick_victim(self, rank_cache, incoming, demand, priority=None):
        best_slot = None
        best_score = None
        for slot, meta in rank_cache.occupied_slots():
            score = (meta.freq, meta.last_used, meta.inserted_at, slot)
            if best_score is None or score < best_score:
                best_score = score
                best_slot = slot
        return best_slot


class DemandAwareEviction:
    """이번 layer 의 demand 가 가장 작은 expert 를 victim.

    Rationale: hot expert 는 어차피 이번 layer 에서도 쓰일 가능성 높으니
    보호.  cold expert (demand 0 또는 낮음) 를 먼저 내림.  tie-break by
    LRU (last_used) → slot.
    """

    name = "demand_aware"

    def pick_victim(self, rank_cache, incoming, demand, priority=None):
        best_slot = None
        best_score = None
        for slot, meta in rank_cache.occupied_slots():
            d = int(demand.get(meta.expert, 0))
            score = (d, meta.last_used, meta.inserted_at, slot)
            if best_score is None or score < best_score:
                best_score = score
                best_slot = slot
        return best_slot


class RandomEviction:
    """Baseline #1: evict a UNIFORMLY RANDOM occupied slot ("위치도 랜덤").

    Seeded deterministically from ``(incoming_layer, incoming_expert, rank,
    layer_seq)`` so the same expert placed on the same rank picks the same victim
    on every EP rank (the cache view is replicated, so every rank's copy of this
    rank's slots is identical -> identical choice).  Pure random by design — the
    dumb baseline; it may even re-evict an expert fetched earlier this layer.
    """

    name = "random"

    def pick_victim(self, rank_cache, incoming, demand, priority=None):
        occ = rank_cache.occupied_slots()
        if not occ:
            return None
        slots = sorted(slot for slot, _ in occ)  # deterministic RNG order
        seed = _stable_seed(
            int(incoming[0]), int(incoming[1]),
            int(rank_cache.rank), int(rank_cache.layer_seq))
        return random.Random(seed).choice(slots)


def _priority_at(priority, key: ExpertKey) -> float:
    """``priority[layer][expert]`` as a float (works for tensor/ndarray/list).

    Out-of-range / missing -> +inf so that expert is the LAST to be evicted
    (we never want to evict something the priority view doesn't cover).
    """
    l, e = key
    try:
        return float(priority[l][e])
    except (IndexError, KeyError, TypeError):
        return float("inf")


class PriorityEviction:
    """EAMC priority 최소 victim.  controller-side EAMC eviction 의 핵심.

    원본(MoE-Infinity ``expert_cache.gpu_evict`` priority 분기)이 Archer/Python
    에서 하던 "lowest priority_score victim" 선택을 controller plan 단계로 이전.
    priority 는 ``priority_aggregator`` 가 만든 [num_layers, num_experts] matrix
    (topo × decoder × frequency, 높을수록 중요).  victim = **최소 priority**.

    원본 ``protected_ondemand`` 의미 보존: 이번 layer 에 demand 가 있는 expert
    (key ∈ demand) 는 곧 compute 되므로 victim 후보에서 제외.  단 모든 occupied
    가 이번 layer demand 면 (capacity 부족) 전체에서 최소 priority 강제 선택.

    priority 미주입(None) 시 LRU 로 graceful fallback — aggregator off / stride
    skip layer 에서 안전 (priority_aggregator docstring 의 'None → LRU' 계약).
    tie-break by (last_used, inserted_at, slot) 로 rank 간 결정성 보장.
    """

    name = "eamc"

    def __init__(self):
        self._lru_fallback = LRUEviction()

    def pick_victim(self, rank_cache, incoming, demand, priority=None):
        if priority is None:
            return self._lru_fallback.pick_victim(rank_cache, incoming, demand)

        occupied = rank_cache.occupied_slots()
        # protect this-layer demanded experts (원본 protected_ondemand)
        candidates = [(s, m) for s, m in occupied if m.expert not in demand]
        if not candidates:                       # all protected -> force over all
            candidates = occupied

        best_slot = None
        best_score = None
        for slot, meta in candidates:
            score = (_priority_at(priority, meta.expert),
                     meta.last_used, meta.inserted_at, slot)
            if best_score is None or score < best_score:
                best_score = score
                best_slot = slot
        return best_slot


_REGISTRY = {
    "lru":           LRUEviction,
    "lfu":           LFUEviction,
    "demand_aware":  DemandAwareEviction,
    "random":        RandomEviction,
    "eamc":          PriorityEviction,
    "priority":      PriorityEviction,   # alias
}


def build_evict_policy(name: Optional[str] = None) -> EvictionPolicy:
    import os
    name = name or os.environ.get("MOE_EP_EVICT_POLICY", "lru")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown evict_policy: {name!r}. Available: {sorted(_REGISTRY)}")
    return cls()
