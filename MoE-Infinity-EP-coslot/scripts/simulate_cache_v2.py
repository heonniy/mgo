#!/usr/bin/env python3
"""Reference simulator v2 — Cooperative offloading (slot + sequential apply
+ union dedup).

Design 참조: /home/work/hyewon.lee/실험/main_exp/moe_cooperative_offloading_design.md

** INDEPENDENT IMPLEMENTATION ** — production controller 의 코드는 일절
import 안 함.  spec 의 §1 결정 + §2 Phase 1/2 알고리즘만 따라 1차 원리로
재구현.  production 과 같은 입력 → 같은 출력 produce 해야 controller spec
이 정확.

Spec (요약):
  Phase 1 — global classification + hit early-kick:
    1.1 per_rank[ep, num_experts]  token count tensor 받음 (routing_log 의
        local_demanded 합)
    1.2 union_demand = (per_rank.any(0)).nonzero()
        hit / miss 분류 (slot cache 의 어느 rank 에 있나)
        miss_set = union_demand - cached  (union dedup, 같은 expert 가
                                            두 rank 에 demand 돼도 1회만)
    1.3 touch hits (last_used = sub-counter)
    1.4 owner_policy.assign_rank(miss_set, demand, cluster_state)
        → {rank: [miss expert keys]}

  Phase 2 — per-rank sequential plan:
    rank 별 demand 내림차순 정렬 후 한 expert 씩 sequential apply:
      victim_slot = pick_victim(rank_cache, ...)
      apply(slot=victim_slot, evict=victim, insert=new)
        ↳ atomic: 물리 배치 + meta.last_used = ++touch_counter

    LRU pick_victim: meta.last_used 최소 slot.

  Phase 4 — verify (skip in simulator).

입력: routing_log glob (production 의 `*_routing_r*.jsonl`)
출력: simulation.json — per_layer (n_hits, n_misses, n_evicts) + totals
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


ExpertKey = Tuple[int, int]


# ============================================================
# slot cache (independent reproduction of RankSlotCache)
# ============================================================
@dataclass
class SimSlotMeta:
    expert: ExpertKey
    inserted_at: int   # layer counter
    last_used: int     # touch_counter (sub-layer monotonic)
    freq: int


class SimRankCache:
    def __init__(self, rank: int, cap: int):
        self.rank = rank
        self.cap = int(cap)
        self.slots: List[Optional[ExpertKey]] = [None] * self.cap
        self.expert_to_slot: Dict[ExpertKey, int] = {}
        self.meta: List[Optional[SimSlotMeta]] = [None] * self.cap
        self.layer_seq = 0
        self.touch_counter = 0  # ★ sub-layer monotonic for last_used

    def is_resident(self, key: ExpertKey) -> bool:
        return key in self.expert_to_slot

    def find_slot(self, key: ExpertKey) -> Optional[int]:
        return self.expert_to_slot.get(key)

    def free_slots(self) -> int:
        return sum(1 for s in self.slots if s is None)

    def first_empty(self) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s is None:
                return i
        return None

    def occupied(self) -> List[Tuple[int, SimSlotMeta]]:
        return [(i, m) for i, m in enumerate(self.meta) if m is not None]

    def apply(self, slot: int, evict: Optional[ExpertKey],
              insert: Optional[ExpertKey], demand_count: int = 0) -> None:
        cur = self.slots[slot]
        if evict is None:
            assert cur is None, f"slot {slot} not empty (has {cur})"
        else:
            assert cur == evict, f"slot {slot} has {cur} != evict {evict}"
            del self.expert_to_slot[evict]
        if insert is None:
            self.slots[slot] = None
            self.meta[slot] = None
            return
        assert insert not in self.expert_to_slot, (
            f"insert {insert} already in slot "
            f"{self.expert_to_slot[insert]}")
        self.slots[slot] = insert
        self.expert_to_slot[insert] = slot
        self.touch_counter += 1
        self.meta[slot] = SimSlotMeta(
            expert=insert, inserted_at=self.layer_seq,
            last_used=self.touch_counter, freq=int(demand_count),
        )

    def touch(self, key: ExpertKey, demand_count: int = 0) -> None:
        slot = self.expert_to_slot.get(key)
        if slot is None:
            return
        m = self.meta[slot]
        if m is None:
            return
        self.touch_counter += 1
        m.last_used = self.touch_counter
        m.freq += int(demand_count)

    def bump_layer(self) -> None:
        self.layer_seq += 1


# ============================================================
# Cluster (multi-rank)
# ============================================================
class SimCluster:
    def __init__(self, ep_size: int, cap_per_rank: int, num_experts: int):
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.per_rank = [SimRankCache(r, cap_per_rank) for r in range(ep_size)]

    def locate(self, key: ExpertKey) -> Optional[int]:
        for r, c in enumerate(self.per_rank):
            if c.is_resident(key):
                return r
        return None


# ============================================================
# Owner policies (independent reproduction)
# ============================================================
def owner_static(miss_keys: List[ExpertKey], demand, ep_size):
    out = {r: [] for r in range(ep_size)}
    for (l, e) in miss_keys:
        out[e % ep_size].append((l, e))
    return out


def owner_balanced(miss_keys: List[ExpertKey], demand, ep_size):
    out = {r: [] for r in range(ep_size)}
    loads = [0] * ep_size
    for key in sorted(miss_keys):
        target = min(range(ep_size), key=lambda r: (loads[r], r))
        out[target].append(key)
        loads[target] += 1
    return out


_OWNERS = {
    "static_placement": owner_static, "static": owner_static, "naive": owner_static,
    "balanced_placement": owner_balanced, "balanced": owner_balanced,
}


# ============================================================
# Eviction policies
# ============================================================
def evict_lru(rank_cache: SimRankCache, incoming: ExpertKey, demand) -> Optional[int]:
    best = None
    best_score = None
    for slot, m in rank_cache.occupied():
        score = (m.last_used, m.inserted_at, slot)
        if best_score is None or score < best_score:
            best_score = score
            best = slot
    return best


def evict_lfu(rank_cache, incoming, demand) -> Optional[int]:
    best = None
    best_score = None
    for slot, m in rank_cache.occupied():
        score = (m.freq, m.last_used, m.inserted_at, slot)
        if best_score is None or score < best_score:
            best_score = score
            best = slot
    return best


_EVICTS = {"lru": evict_lru, "lfu": evict_lfu}


# ============================================================
# RankPlanner: per-rank sequential plan
# ============================================================
def plan_for_rank(rank, miss_experts, rank_cache, demand, evict_fn):
    """Returns list of (expert, dst_slot, victim) tuples (sequential applied)."""
    # demand 내림차순, tie-break by (layer, expert)
    ordered = sorted(
        miss_experts,
        key=lambda k: (-int(demand.get(k, 0)), k[0], k[1]),
    )
    ops = []
    for expert in ordered:
        if rank_cache.is_resident(expert):
            continue
        empty = rank_cache.first_empty()
        if empty is not None:
            dst = empty
            victim = None
        else:
            v = evict_fn(rank_cache, incoming=expert, demand=demand)
            if v is None:
                # cap=0 또는 모든 slot 비어있음 (impossible if cap>0).  drop.
                continue
            dst = v
            victim = rank_cache.slots[dst]
        ops.append((expert, dst, victim))
        rank_cache.apply(
            slot=dst, evict=victim, insert=expert,
            demand_count=int(demand.get(expert, 0)),
        )
    return ops


# ============================================================
# load routing_log
# ============================================================
def load_routing_log(glob_pat: str) -> List[dict]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise SystemExit(f"no routing log files match: {glob_pat}")
    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    records.sort(
        key=lambda r: (int(r["step"]), int(r["layer"]), int(r["ep_rank"])))
    return records


def group_by_step_layer(records):
    """Yield (step, layer, [records...])."""
    cur = None
    bucket = []
    for r in records:
        key = (int(r["step"]), int(r["layer"]))
        if cur is None:
            cur = key
        if key != cur:
            yield cur[0], cur[1], bucket
            bucket = []
            cur = key
        bucket.append(r)
    if bucket:
        yield cur[0], cur[1], bucket


# ============================================================
# main simulate
# ============================================================
def simulate(records, ep_size, cap, num_experts, owner_name, evict_name):
    if owner_name not in _OWNERS:
        raise SystemExit(f"unknown owner: {owner_name}")
    if evict_name not in _EVICTS:
        raise SystemExit(f"unknown evict: {evict_name}")
    owner_fn = _OWNERS[owner_name]
    evict_fn = _EVICTS[evict_name]
    INF = cap == 0

    # cap=∞ 면 slots list 너무 커서 OOM.  num_layers × num_experts 정도면 충분
    # — 한 forward 의 모든 demand 수용 가능.
    cap_eff = (num_experts * 200) if INF else cap  # 200 layer 정도 ceiling
    cluster = SimCluster(ep_size, cap_eff, num_experts)
    per_layer = []
    totals = {"hits": 0, "misses": 0, "evictions": 0, "drops": 0}

    for step, layer, bucket in group_by_step_layer(records):
        # Phase 1.1 — global demand 합산
        global_count = [0] * num_experts
        for r in bucket:
            for e in r.get("local_demanded", []) or []:
                global_count[int(e)] += 1  # ← per_rank.sum 의 근사 (count 정보가
                # routing_log 에 없으면 1 token / expert 가정).  production
                # 의 demand_count 와 차이가 있을 수 있지만 hits/misses/evicts
                # count 자체엔 영향 없음 (sort tie-break 만 영향).
        demanded_ids = [e for e in range(num_experts) if global_count[e] > 0]
        demand: Dict[ExpertKey, int] = {
            (layer, e): global_count[e] for e in demanded_ids
        }

        # Phase 1.2 — hit / miss + union dedup
        hit_keys, miss_keys = set(), set()
        for e in demanded_ids:
            key = (layer, e)
            r = cluster.locate(key)
            if r is None:
                miss_keys.add(key)
            else:
                hit_keys.add(key)

        # Phase 1.3 — touch hits
        for key in hit_keys:
            r = cluster.locate(key)
            cluster.per_rank[r].touch(key, demand_count=demand.get(key, 0))

        # Phase 1.4 — owner_policy.assign_rank (union dedup miss_keys)
        miss_per_rank = owner_fn(sorted(miss_keys), demand, ep_size)

        # Phase 2 — per-rank sequential plan
        all_ops = []
        for r in range(ep_size):
            ops = plan_for_rank(
                rank=r, miss_experts=miss_per_rank.get(r, []),
                rank_cache=cluster.per_rank[r], demand=demand,
                evict_fn=evict_fn,
            )
            all_ops.extend(ops)

        # 통계
        n_hits = len(hit_keys)
        n_misses = len(all_ops)
        n_evicts = sum(1 for (_, _, v) in all_ops if v is not None)
        n_drops = len(miss_keys) - n_misses

        # per-rank distributions (owner_policy 결과 + classify 결과 직접)
        miss_dist = [len(miss_per_rank.get(r, [])) for r in range(ep_size)]
        hit_dist = [0] * ep_size
        # hit 의 owner_rank 는 touch 전 cluster.locate 가 가리킨 rank.  단,
        # Phase 1.3 의 touch_use 가 last_used 만 갱신하지 rank 이동 X.
        # 그러나 우리는 위에서 이미 touch 후라 cluster 상태가 변경 — 안전하게
        # hit_keys 의 각 key 에 대해 다시 locate.
        for key in hit_keys:
            r = cluster.locate(key)
            if r is not None and 0 <= r < ep_size:
                hit_dist[r] += 1

        per_layer.append({
            "step": step,
            "layer": layer,
            "n_demanded": len(demanded_ids),
            "n_hits": n_hits,
            "n_misses": n_misses,
            "n_evicts": n_evicts,
            "n_drops": n_drops,
            "hits_per_serving_rank": hit_dist,
            "miss_target_rank_dist": miss_dist,
            "cache_size_per_rank": [
                c.cap - c.free_slots() for c in cluster.per_rank],
        })
        totals["hits"] += n_hits
        totals["misses"] += n_misses
        totals["evictions"] += n_evicts
        totals["drops"] += n_drops

        for c in cluster.per_rank:
            c.bump_layer()

    return {
        "ep_size": ep_size,
        "cap_per_rank": cap if not INF else "unlimited",
        "owner_policy": owner_name,
        "evict_policy": evict_name,
        "n_layers_simulated": len(per_layer),
        "totals": totals,
        "per_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing-glob", required=True)
    ap.add_argument("--ep-size", type=int, required=True)
    ap.add_argument("--num-experts", type=int, required=True)
    ap.add_argument("--cap", type=int, required=True, help="0 = unlimited")
    ap.add_argument("--owner", default="static_placement",
                    choices=list(_OWNERS.keys()))
    ap.add_argument("--evict", default="lru", choices=list(_EVICTS.keys()))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_routing_log(args.routing_glob)
    print(f"[sim_v2] loaded {len(records)} routing records", flush=True)
    result = simulate(
        records, args.ep_size, args.cap, args.num_experts,
        args.owner, args.evict)
    Path(args.out).write_text(json.dumps(result, indent=2))
    t = result["totals"]
    print(f"[sim_v2] → {args.out}")
    print(f"[sim_v2] totals: hits={t['hits']} misses={t['misses']} "
          f"evictions={t['evictions']} drops={t['drops']}")


if __name__ == "__main__":
    main()
