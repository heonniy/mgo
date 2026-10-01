"""Independent OURS oracle — re-implements algorithm.md §1-4 from the spec ONLY.

Mirrors the production ``OursPlacementPlanner`` *orchestration* with independent
code (own candidate-building, DP, Hungarian, emit/apply) so a differential test
catches integration/shadow-apply/ordering bugs.  Uses numpy/scipy directly and
reuses ``reference.RefSlotCache`` (import-pure) for slot semantics — it imports
NOTHING from ``moe_infinity_ep``.

Shapes/normalisation match the spec:
  * prefill benefit = raw D_prefill (algorithm.md §1.3); Hungarian capacity K.
  * decode: per-GPU LFU-ordered candidates (empty first @0), prefix DP for q_g,
    then Hungarian on normalised CurrentLocality (§3.1).  evict cost = freq stub.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from reference import RefSlotCache

ExpertKey = Tuple[int, int]


# ---- independent solver primitives (same spec as assignment.py) ----
def _ck(M: int, G: int) -> int:
    return math.ceil(M / G) if M > 0 else 0


def _hungarian(D: np.ndarray, caps: np.ndarray) -> np.ndarray:
    M, G = D.shape
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    slot_gpu = np.repeat(np.arange(G), caps)
    cost = D.max(axis=1, keepdims=True) - D[:, slot_gpu]
    rows, cols = linear_sum_assignment(cost)
    a = np.full(M, -1, dtype=np.int64)
    a[rows] = slot_gpu[cols]
    return a


def _match(B: np.ndarray) -> np.ndarray:
    M = B.shape[0]
    if M == 0:
        return np.zeros(0, dtype=np.int64)
    cost = B.max(axis=1, keepdims=True) - B
    rows, cols = linear_sum_assignment(cost)
    m = np.full(M, -1, dtype=np.int64)
    m[rows] = cols
    return m


def _quota_dp(prefix: List[List[float]], M: int) -> Optional[List[int]]:
    G = len(prefix)
    qmax = [len(p) - 1 for p in prefix]
    if sum(qmax) < M:
        return None
    INF = float("inf")
    dp = [INF] * (M + 1)
    dp[0] = 0.0
    choice = [[-1] * (M + 1) for _ in range(G)]
    for i in range(G):
        ndp = [INF] * (M + 1)
        pc = prefix[i]
        for m in range(M + 1):
            if dp[m] == INF:
                continue
            for q in range(0, min(qmax[i], M - m) + 1):
                c = dp[m] + pc[q]
                if c < ndp[m + q]:
                    ndp[m + q] = c
                    choice[i][m + q] = q
        dp = ndp
    if dp[M] == INF:
        return None
    q = [0] * G
    m = M
    for i in range(G - 1, -1, -1):
        q[i] = choice[i][m]
        m -= q[i]
    return q


class OursRefController:
    def __init__(self, ep_size: int, cap_per_rank: int, num_experts: int):
        self.ep_size = ep_size
        self.cap = cap_per_rank
        self.num_experts = num_experts
        self.per_rank = [RefSlotCache(r, cap_per_rank) for r in range(ep_size)]

    # ---- global view ----
    def locate(self, key: ExpertKey) -> Optional[int]:
        for r in range(self.ep_size):
            if self.per_rank[r].is_resident(key):
                return r
        return None

    def shadow(self) -> Dict[int, List[Optional[ExpertKey]]]:
        return {r: list(self.per_rank[r].slots) for r in range(self.ep_size)}

    def seed(self, shadow: Dict[int, List[Optional[ExpertKey]]]) -> None:
        for r, slots in shadow.items():
            rc = self.per_rank[int(r)]
            for i, cell in enumerate(slots):
                if cell is not None:
                    rc.apply(slot=i, evict=None,
                             insert=(int(cell[0]), int(cell[1])), demand_count=0)

    def bump(self) -> None:
        for rc in self.per_rank:
            rc.bump_layer()

    # ---- helpers ----
    def _D(self, miss: List[ExpertKey], per_rank_demand) -> np.ndarray:
        G = self.ep_size
        ge = np.asarray(per_rank_demand, dtype=np.int64)  # [G, E]
        ids = [e for (_l, e) in miss]
        return ge[:, ids].T.astype(np.int64)

    def _fit(self, miss, D, layer_id):
        total_phys = sum(self.per_rank[g].cap for g in range(self.ep_size))
        M = len(miss)
        if M <= total_phys:
            return miss, D
        rowsum = D.sum(axis=1)
        order = sorted(range(M),
                       key=lambda i: (-int(rowsum[i]), miss[i][0], miss[i][1]))
        keep = sorted(order[:total_phys])
        return [miss[i] for i in keep], D[keep]

    # ---- one full layer ----
    def step(self, per_rank_demand: List[List[int]], layer_id: int = 0,
             is_decode: bool = False) -> dict:
        G = self.ep_size
        # Phase 1: global demand union
        global_count = [0] * self.num_experts
        for r in range(G):
            for e in range(self.num_experts):
                global_count[e] += int(per_rank_demand[r][e])
        demanded = [e for e in range(self.num_experts) if global_count[e] > 0]
        demand = {(layer_id, e): global_count[e] for e in demanded}

        hit_ops = []
        miss_set: List[ExpertKey] = []
        for e in demanded:
            key = (layer_id, e)
            owner = self.locate(key)
            if owner is None:
                miss_set.append(key)
            else:
                slot = self.per_rank[owner].find_slot(key)
                hit_ops.append({"expert": key, "owner_rank": owner, "slot": slot})
        for op in hit_ops:
            self.per_rank[op["owner_rank"]].touch_use(
                op["expert"], demand_count=demand.get(op["expert"], 0))

        miss = sorted(miss_set)
        if is_decode:
            fetch_ops = self._plan_decode(miss, demand, per_rank_demand, layer_id)
        else:
            fetch_ops = self._plan_prefill(miss, demand, per_rank_demand, layer_id)

        routing_map_miss = {op["expert"]: op["fetcher_rank"] for op in fetch_ops}
        expert_to_rank = {op["expert"]: op["owner_rank"] for op in hit_ops}
        expert_to_rank.update(routing_map_miss)
        fetch_per_rank = {r: sorted(op["expert"] for op in fetch_ops
                                    if op["fetcher_rank"] == r) for r in range(G)}
        return {
            "demand": demand,
            "hit_ops": hit_ops,
            "misses": sorted(miss_set),
            "fetch_ops": fetch_ops,
            "fetch_per_rank": fetch_per_rank,
            "routing_map": expert_to_rank,
            "shadow_after": self.shadow(),
        }

    # ---- prefill (algorithm.md §1) ----
    def _plan_prefill(self, miss, demand, per_rank_demand, layer_id):
        G = self.ep_size
        if not miss:
            return []
        D = self._D(miss, per_rank_demand)
        miss, D = self._fit(miss, D, layer_id)
        M = len(miss)
        if M == 0:
            return []
        K = _ck(M, G)
        caps = np.array([min(K, self.per_rank[g].cap) for g in range(G)],
                        dtype=np.int64)
        assign = _hungarian(D, caps)
        by_gpu = {g: [] for g in range(G)}
        for i, g in enumerate(assign):
            by_gpu[int(g)].append(i)
        ops = []
        for g in range(G):
            rows = by_gpu[g]
            if not rows:
                continue
            rc = self.per_rank[g]
            # target slots chosen UP FRONT from pre-layer state: empties first,
            # then LFU-coldest pre-layer residents (no self-eviction).
            n = len(rows)
            targets = []
            empties = [s for s in range(rc.cap) if rc.slots[s] is None]
            for s in empties[:n]:
                targets.append((s, None))
            remaining = n - len(targets)
            if remaining > 0:
                occ = sorted(rc.occupied_slots(),
                             key=lambda sm: (sm[1].freq, sm[1].last_used,
                                             sm[1].inserted_at, sm[0]))
                for slot, meta in occ[:remaining]:
                    targets.append((slot, meta.expert))
            rows.sort(key=lambda i: (-int(D[i, g]), miss[i][0], miss[i][1]))
            order = 0
            for (dst, victim), i in zip(targets, rows):
                expert = miss[i]
                ops.append({"expert": expert, "fetcher_rank": g, "dst_slot": dst,
                            "victim_expert": victim, "order": order})
                order += 1
                rc.apply(slot=dst, evict=victim, insert=expert,
                         demand_count=int(demand.get(expert, 0)))
        return ops

    # ---- decode (algorithm.md §2-3) ----
    def _plan_decode(self, miss, demand, per_rank_demand, layer_id):
        G = self.ep_size
        if not miss:
            return []
        D = self._D(miss, per_rank_demand)
        miss, D = self._fit(miss, D, layer_id)
        M = len(miss)
        if M == 0:
            return []
        K = _ck(M, G)
        # D1: candidate lists
        cand = []
        for g in range(G):
            rc = self.per_rank[g]
            lst = []
            for s in range(rc.cap):
                if rc.slots[s] is None:
                    lst.append((s, 0.0, None))
            occ = sorted(rc.occupied_slots(),
                         key=lambda sm: (sm[1].freq, sm[1].last_used,
                                         sm[1].inserted_at, sm[0]))
            for slot, meta in occ:
                lst.append((slot, float(meta.freq), meta.expert))
            cand.append(lst[:min(K, len(lst))])
        # D2: prefix + DP
        prefix = []
        for g in range(G):
            pc = [0.0]
            for (_s, c, _v) in cand[g]:
                pc.append(pc[-1] + c)
            prefix.append(pc)
        q_g = _quota_dp(prefix, M)
        assert q_g is not None
        selected = []
        for g in range(G):
            for j in range(q_g[g]):
                s, _c, v = cand[g][j]
                selected.append((g, s, v))
        # D3: benefit (normalised current locality) + Hungarian
        S = D.sum(axis=1, keepdims=True).astype(np.float64)
        current_loc = D.astype(np.float64) / S
        gpu_cols = np.array([g for (g, _s, _v) in selected], dtype=np.int64)
        B = current_loc[:, gpu_cols]
        match = _match(B)
        # D4: group, order by D_now desc, emit + apply
        by_gpu = {g: [] for g in range(G)}
        for i in range(M):
            g, slot, victim = selected[int(match[i])]
            by_gpu[g].append((i, slot, victim))
        ops = []
        for g in range(G):
            items = by_gpu[g]
            if not items:
                continue
            items.sort(key=lambda t: (-int(D[t[0], g]), miss[t[0]][0], miss[t[0]][1]))
            rc = self.per_rank[g]
            order = 0
            for (i, slot, victim) in items:
                expert = miss[i]
                ops.append({"expert": expert, "fetcher_rank": g, "dst_slot": slot,
                            "victim_expert": victim, "order": order})
                order += 1
                rc.apply(slot=slot, evict=victim, insert=expert,
                         demand_count=int(demand.get(expert, 0)))
        return ops
