"""OwnerPolicy — Phase 1 의 miss → rank 분배 정책.

새 architecture 에서 placement 의 역할이 둘로 쪼개진다:

  1. ``OwnerPolicy.assign_rank`` (Phase 1, 전역 결정)
     — 어느 miss expert 를 어느 rank 가 책임지고 fetch 할지만 결정.
     — slot 위치 결정 NOT.  rank 분배만.
     — 결정성 (모든 rank 가 같은 입력 → 같은 출력) 필수.

  2. ``RankPlanner.pick_victim`` (Phase 2, per-rank 결정)
     — 그 rank 가 받은 miss 들을 자기 slot 에 어떻게 박을지 sequential 결정.
     — owner_policy 와 분리 (이 파일에는 없음, eviction_policy 의 영역).

Policy 들은 plug-in:
  * NaiveOwner    — ``expert_id % ep_size`` (locality preserving)
  * BalancedOwner — ``argmin(rank_load)`` + tie-break (load balancing)

OwnerPolicy 가 보는 것:
  * miss_keys      : 이번 layer 의 miss (전역)
  * demand[E]      : expert 별 token 수 (local + remote)
  * cluster_state  : GlobalSlotCacheView (rank 별 점유 상태 등)

OwnerPolicy 가 결정하는 것:
  * dict[rank, list[ExpertKey]]  ─ 그 rank 가 책임질 miss 들의 정렬되지 않은 집합
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Protocol, Tuple

from .slot_cache import ExpertKey, GlobalSlotCacheView


def _stable_seed(*vals: int) -> int:
    """Process-independent deterministic seed from ints.

    Python's builtin ``hash`` is salted per-process (PYTHONHASHSEED), so it must
    NOT be used for cross-rank determinism — every rank would seed differently
    and the resulting plans would diverge (Phase-4 drift FATAL).  This mix is
    stable across processes and ranks.
    """
    s = 0
    for v in vals:
        s = (s * 1000003 + int(v)) & 0xFFFFFFFF
    return s


class OwnerPolicy(Protocol):
    """rank 분배 정책 protocol.  결정은 deterministic."""
    name: str

    def assign_rank(
        self,
        miss_keys: List[ExpertKey],
        demand: Dict[ExpertKey, int],
        cluster_state: GlobalSlotCacheView,
    ) -> Dict[int, List[ExpertKey]]:
        ...


class StaticPlacement:
    """각 expert 가 **고정된 home rank** 를 가진다 (=`expert_id % ep_size`).

    miss expert 는 항상 자기 home rank 가 책임.

    장점:
        * routing 이 결정론적이라 단순/예측 가능
        * 같은 expert 가 항상 같은 rank 로 모임 → cache locality 좋음
        * owner 결정에 전역 load 계산 불필요 (overhead 최소)
    단점:
        * demand skew 시 hot expert 의 home rank 만 과부하
    용도: baseline, demand 균등하거나 placement 미리 최적화된 경우
    """
    name = "static_placement"

    def assign_rank(
        self,
        miss_keys: List[ExpertKey],
        demand: Dict[ExpertKey, int],
        cluster_state: GlobalSlotCacheView,
    ) -> Dict[int, List[ExpertKey]]:
        ep_size = cluster_state.ep_size
        out: Dict[int, List[ExpertKey]] = {r: [] for r in range(ep_size)}
        for (l, e) in miss_keys:
            out[e % ep_size].append((l, e))
        return out


class BalancedPlacement:
    """rank 별 fetch expert COUNT 만 균등(±1), 배정 자체는 seeded RANDOM.

    정의 (2026-06-10 확정): balanced = "per-rank fetch 개수 균등 + 그 외 무작위".
    demand(토큰 수)·expert id 어느 쪽과도 상관이 없도록, miss 를 seeded shuffle
    한 뒤 random-offset round-robin 으로 나눈다.  (이전 구현은 id-순 greedy
    count 빈패킹이라 expert id ↔ rank 상관이 있었고 docstring 은 토큰 가중을
    잘못 주장했음 — 둘 다 제거.  토큰-가중 LPT 변종(DemandAwareOwner)도 같은
    날짜에 삭제: baseline 에 수요 정보가 새어들지 않게.)

    결정성: RandomOwner 와 동일하게 (base_seed, layer_id, layer_seq) seed →
    모든 EP rank 가 동일 배정 (drift-safe), 같은 seed = run 간 재현.

    장점: 개수 균형 보장 + 어떤 수요 정보도 안 씀 (순수 baseline)
    단점: locality 무시 (같은 expert 가 layer 마다 다른 rank)
    """
    name = "balanced_placement"

    def __init__(self) -> None:
        import os
        self.base_seed = int(os.environ.get("MOE_EP_RANDOM_SEED", "0"))

    def assign_rank(
        self,
        miss_keys: List[ExpertKey],
        demand: Dict[ExpertKey, int],
        cluster_state: GlobalSlotCacheView,
    ) -> Dict[int, List[ExpertKey]]:
        ep_size = cluster_state.ep_size
        out: Dict[int, List[ExpertKey]] = {r: [] for r in range(ep_size)}
        keys = sorted(miss_keys)   # deterministic RNG consumption order
        if not keys:
            return out
        layer_id = keys[0][0]
        layer_seq = cluster_state.per_rank[0].layer_seq
        rng = random.Random(_stable_seed(self.base_seed, layer_id, layer_seq))
        rng.shuffle(keys)                       # random membership
        off = rng.randrange(ep_size)            # random quota remainder
        for i, key in enumerate(keys):          # round-robin → counts ±1
            out[(i + off) % ep_size].append(key)
        return out


class RandomOwner:
    """Baseline #1: each miss expert -> an INDEPENDENT UNIFORMLY RANDOM rank, with
    NO per-GPU quota (so per-rank counts come out lumpy/unbalanced — the dumb
    baseline; "gpu당 quota든 위치든 그냥 랜덤").

    A rank may be assigned more misses than it has free slots; the per-rank planner
    then evicts (random victim) to fit, which can thrash — that is intended for the
    no-placement reference.  Randomness is seeded deterministically from
    ``(layer_id, layer_seq)`` (via :func:`_stable_seed`, NOT builtin hash) so every
    EP rank computes the identical assignment and the replicated shadow stays
    byte-identical.  ``layer_seq`` is read from the replicated cache view (all ranks
    bump in lockstep).  An experiment-level base seed (``MOE_EP_RANDOM_SEED``) is
    folded in so the placement is REPRODUCIBLE (same seed -> identical assignment
    every run) and CONTROLLABLE (different seed -> different placement).
    """
    name = "random"

    def __init__(self) -> None:
        import os
        self.base_seed = int(os.environ.get("MOE_EP_RANDOM_SEED", "0"))

    def assign_rank(
        self,
        miss_keys: List[ExpertKey],
        demand: Dict[ExpertKey, int],
        cluster_state: GlobalSlotCacheView,
    ) -> Dict[int, List[ExpertKey]]:
        ep_size = cluster_state.ep_size
        out: Dict[int, List[ExpertKey]] = {r: [] for r in range(ep_size)}
        keys = sorted(miss_keys)  # deterministic RNG consumption order
        if not keys:
            return out
        layer_id = keys[0][0]
        layer_seq = cluster_state.per_rank[0].layer_seq
        rng = random.Random(_stable_seed(self.base_seed, layer_id, layer_seq))
        for key in keys:
            out[rng.randrange(ep_size)].append(key)   # no quota — fully random
        return out


_REGISTRY: Dict[str, type] = {
    # design 문서 §3 의 정식 이름
    "static_placement":    StaticPlacement,
    "balanced_placement":  BalancedPlacement,
    # 짧은 alias (YAML / env 편의)
    "static":              StaticPlacement,
    "balanced":            BalancedPlacement,
    "random":              RandomOwner,
    # legacy alias (이전 코드 호환)
    "naive":               StaticPlacement,
    "naive_owner":         StaticPlacement,
    "balanced_owner":      BalancedPlacement,
}


def build_owner_policy(name: Optional[str] = None) -> OwnerPolicy:
    import os
    name = name or os.environ.get("MOE_EP_OWNER_POLICY", "static_placement")
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown owner_policy: {name!r}. Available: {sorted(_REGISTRY)}")
    return cls()
