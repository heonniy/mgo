"""OursPlacementPlanner — controller-side JOINT placement (ownership + slot +
fetch order decided together).  Baseline #4 ("OURS").

Design: /home/work/hyewon.lee/실험/placement 구현 실험/algorithm.md

Unlike the baseline two-step (``owner_policy.assign_rank`` then per-rank
``rank_planner.plan``), this planner sees the full per-rank demand matrix + every
rank's cache shadow at once and produces the complete ``List[FetchOp]`` for the
layer, deciding for each miss expert WHICH rank fetches it, INTO which physical
slot (evicting which victim), and in WHICH order — jointly.

It is a drop-in replacement for the per-rank loop inside
``GlobalCacheController.plan_misses``: it mutates the replicated shadow via
``rank_cache.apply`` exactly like ``RankPlanner`` (so Phase-4 verify drift stays
0) and emits FetchOps with contiguous per-rank ``order`` (so the archer PlanQueue
contract holds).

PREFILL (algorithm.md §1): minimise current-token / NVLink movement.
  D_prefill(e,g) = per_rank_count[g][e].  Expanded-slot Hungarian (capacity
  K=ceil(M/G)) assigns expert->GPU maximising local demand served.  Per GPU:
  empty slots first, else LFU victim; fetch order = D_prefill(e,g) desc.

DECODE (algorithm.md §2-3): eviction-cost quota DP + locality/future Hungarian.
  Per GPU build an ORDERED physical candidate list (empty slots first @ cost 0,
  then occupied in LFU order @ cost = importance), prefix-cost DP picks q_g
  (sum=M), then Hungarian matches misses to the selected physical slots
  maximising CurrentLocality + lambda*FutureReuseLocality; fetch order = D_now desc.

Stage1: ``evict_cost_provider`` / ``future_affinity_provider`` are None ->
eviction cost falls back to LFU ``meta.freq`` and the future term is 0 (lambda
default 0).  Stage2 injects the live A2 providers.

Determinism: every EP rank runs this on byte-identical inputs (all-gathered
demand, replicated cache view) -> byte-identical FetchOps/shadow.  All tie-breaks
are fixed (see :mod:`assignment` and the per-GPU sorts below).
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import assignment as A
from .layer_plan import ExpertKey, FetchOp
from .slot_cache import GlobalSlotCacheView, RankSlotCache

_HEAVY = os.environ.get("MOE_EP_HEAVY_DEBUG", "0") == "1"


def _to_np_int(per_rank_count) -> np.ndarray:
    """[ep_size, num_experts] tensor/array -> int64 ndarray."""
    if per_rank_count is None:
        raise ValueError("OURS planner requires per_rank_count (demand matrix)")
    if hasattr(per_rank_count, "detach"):
        return per_rank_count.detach().cpu().numpy().astype(np.int64)
    return np.asarray(per_rank_count).astype(np.int64)


class OursPlacementPlanner:
    def __init__(
        self,
        ep_size: int,
        future_lambda: float = 0.0,
        evict_cost_provider: Optional[Callable[[ExpertKey], float]] = None,
        future_affinity_provider: Optional[Callable[[ExpertKey, int], float]] = None,
    ):
        self.ep_size = int(ep_size)
        self.future_lambda = float(future_lambda)
        self.evict_cost_provider = evict_cost_provider
        self.future_affinity_provider = future_affinity_provider
        # ablation: drop the locality placement (Hungarian) while keeping quota DP
        # + eviction identical -> isolates the placement(locality) effect.
        self.no_placement = os.environ.get("MOE_EP_OURS_NO_PLACEMENT", "0") == "1"
        # ablation: stable-home GREEDY placement instead of exact Hungarian — each
        # miss claims its highest-demand GPU/slot still free (deterministic, integer
        # tie-breaks) -> isolates the optimality of the assignment from its locality.
        self.greedy = os.environ.get("MOE_EP_OURS_GREEDY", "0") == "1"
        # ablation: decode placement mirrors PREFILL — cap-only Hungarian
        # (caps=K per GPU) over D_now, NO eviction-cost quota DP -> isolates the
        # quota DP's contribution from the locality assignment.
        self.decode_caponly = os.environ.get("MOE_EP_OURS_DECODE_CAPONLY", "0") == "1"

    # ===================================================================
    # entry
    # ===================================================================
    def plan_layer(
        self,
        miss_set: List[ExpertKey],
        demand: Dict[ExpertKey, int],
        per_rank_count,
        cache_view: GlobalSlotCacheView,
        layer_id: int,
        is_decode: bool,
    ) -> List[FetchOp]:
        miss = sorted(miss_set)
        if not miss:
            return []
        ge = _to_np_int(per_rank_count)
        if is_decode:
            return self._plan_decode(miss, demand, ge, cache_view, layer_id)
        return self._plan_prefill(miss, demand, ge, cache_view, layer_id)

    # ===================================================================
    # helpers
    # ===================================================================
    def _build_D(self, miss: List[ExpertKey], ge: np.ndarray) -> np.ndarray:
        """[M, G] integer demand D[i,g] = tokens on GPU g routing to miss[i]."""
        expert_ids = [e for (_l, e) in miss]
        return ge[:, expert_ids].T.astype(np.int64)  # [M, G]

    def _fit_capacity(
        self, miss: List[ExpertKey], D: np.ndarray, total_phys: int, layer_id: int,
    ) -> Tuple[List[ExpertKey], np.ndarray]:
        """Drop lowest-total-demand misses if M exceeds total physical capacity.

        Only triggers in the degenerate regime where the cache literally cannot
        hold the layer's miss set (sum of per-rank caps < M).  In realistic
        configs cap >> K so this never fires.  We never silently truncate without
        logging.
        """
        M = len(miss)
        if M <= total_phys:
            return miss, D
        keep = total_phys
        # keep highest total-demand misses; tie by (layer, expert) asc
        rowsum = D.sum(axis=1)
        order = sorted(range(M), key=lambda i: (-int(rowsum[i]), miss[i][0], miss[i][1]))
        keep_idx = sorted(order[:keep])
        dropped = M - keep
        print(f"[OURS planner] L{layer_id} OVER-SUBSCRIBED: M={M} > total_phys="
              f"{total_phys}; dropping {dropped} lowest-demand misses (cache too "
              f"small for working set).", flush=True)
        miss2 = [miss[i] for i in keep_idx]
        D2 = D[keep_idx]
        return miss2, D2

    def _evict_cost(self, meta) -> float:
        """Importance cost of evicting this resident (algorithm.md §2.1).

        free slot handled by caller (cost 0).  Stage2: A2 hotness via
        ``evict_cost_provider`` (hot -> high cost -> protected).  Stage1: LFU
        ``meta.freq`` as a stub so colder (less-used) residents cost less.
        """
        if self.evict_cost_provider is not None:
            return float(self.evict_cost_provider(meta.expert))
        return float(meta.freq)

    # ===================================================================
    # PREFILL
    # ===================================================================
    def _plan_prefill(
        self, miss: List[ExpertKey], demand: Dict[ExpertKey, int],
        ge: np.ndarray, cache_view: GlobalSlotCacheView, layer_id: int,
    ) -> List[FetchOp]:
        G = self.ep_size
        D = self._build_D(miss, ge)  # [M, G]
        total_phys = sum(cache_view.per_rank[g].cap for g in range(G))
        miss, D = self._fit_capacity(miss, D, total_phys, layer_id)
        M = len(miss)
        if M == 0:
            return []
        K = A.compute_k(M, G)
        # effective per-GPU capacity for this layer = min(K, physical cap).
        caps = np.array([min(K, cache_view.per_rank[g].cap) for g in range(G)],
                        dtype=np.int64)
        if self.no_placement:
            assign = np.array([i % G for i in range(M)], dtype=np.int64)  # round-robin (no locality)
        elif self.greedy:
            assign = self._greedy_assign(miss, D, caps)  # stable-home greedy
        else:
            assign = A.solve_hungarian(D, caps=caps)  # expert_row -> GPU

        by_gpu: Dict[int, List[int]] = {g: [] for g in range(G)}
        for i, g in enumerate(assign):
            by_gpu[int(g)].append(i)

        fetch_ops: List[FetchOp] = []
        for g in range(G):
            rows = by_gpu[g]
            if not rows:
                continue
            rc = cache_view.per_rank[g]
            # Pick the n target slots UP FRONT from the PRE-LAYER state: empty
            # slots first (victim None), then the n_evict LFU-coldest *pre-layer*
            # residents.  Selecting before any apply guarantees (a) distinct
            # dst_slots and (b) an expert fetched this layer is never chosen as a
            # victim of another fetch (no self-eviction / wasted fetch).
            targets = self._prefill_targets(rc, len(rows))
            # §1.8: same-GPU fetch order = D_prefill(e,g) desc, tie (layer,expert)
            rows.sort(key=lambda i: (-int(D[i, g]), miss[i][0], miss[i][1]))
            order = 0
            for (dst, victim), i in zip(targets, rows):
                expert = miss[i]
                fetch_ops.append(FetchOp(
                    expert=expert, fetcher_rank=g, dst_slot=dst,
                    victim_expert=victim, order=order, source="pcie"))
                order += 1
                rc.apply(slot=dst, evict=victim, insert=expert,
                         demand_count=int(demand.get(expert, 0)))
        return fetch_ops

    def _prefill_targets(
        self, rc: RankSlotCache, n: int,
    ) -> List[Tuple[int, Optional[ExpertKey]]]:
        """n distinct (slot, victim) targets from PRE-LAYER state: empties first
        (cost-free), then n_evict LFU-coldest residents.  ``n <= cap`` (Hungarian
        capacity <= cap), so there are always enough slots."""
        targets: List[Tuple[int, Optional[ExpertKey]]] = []
        empties = [s for s in range(rc.cap) if rc.slots[s] is None]
        for s in empties[:n]:
            targets.append((s, None))
        remaining = n - len(targets)
        if remaining > 0:
            occ = sorted(
                rc.occupied_slots(),
                key=lambda sm: (sm[1].freq, sm[1].last_used, sm[1].inserted_at, sm[0]),
            )
            for slot, meta in occ[:remaining]:
                targets.append((slot, meta.expert))
        return targets

    def _greedy_assign(
        self, miss: List[ExpertKey], D: np.ndarray, caps: np.ndarray,
    ) -> np.ndarray:
        """Stable-home greedy expert->GPU assignment (drop-in for solve_hungarian).

        Each miss claims its highest-local-demand GPU with capacity left — a
        local heuristic, NOT the global Hungarian optimum.  Misses are processed
        in descending max-over-GPU demand (tie: ascending expert id) so the most
        locality-sensitive experts grab their home GPU first.  All tie-breaks are
        fixed (lowest GPU index) -> byte-identical across EP ranks (drift=0)."""
        M, G = D.shape
        rem = caps.astype(np.int64).copy()
        assign = np.full(M, -1, dtype=np.int64)
        order = sorted(
            range(M),
            key=lambda i: (-int(D[i].max()), miss[i][0], miss[i][1]))
        for i in order:
            best_g, best_d = -1, None
            for g in range(G):  # ascending g => lowest GPU index wins ties
                if rem[g] <= 0:
                    continue
                d = int(D[i, g])
                if best_g < 0 or d > best_d:
                    best_g, best_d = g, d
            assert best_g >= 0, "greedy_assign ran out of capacity (M > sum caps)"
            assign[i] = best_g
            rem[best_g] -= 1
        return assign

    def _greedy_match(
        self, miss: List[ExpertKey], benefit: np.ndarray,
    ) -> np.ndarray:
        """Stable-home greedy expert->selected-column bijection (drop-in for
        decode_match).

        Each miss claims its highest-benefit selected slot that is still free —
        a local heuristic, NOT the global Hungarian optimum.  Misses are processed
        in descending max-over-column benefit (tie: ascending expert id); within a
        row the lowest free column wins on equal benefit.  Because every column is
        claimed exactly once, the result is a valid bijection over the M selected
        columns (so the per-rank q_g quotas the columns encode are respected) and
        is byte-identical across EP ranks (drift=0)."""
        M = benefit.shape[0]
        free = [True] * M
        match = np.full(M, -1, dtype=np.int64)
        rowmax = benefit.max(axis=1) if M else np.zeros(0)
        order = sorted(
            range(M),
            key=lambda i: (-float(rowmax[i]), miss[i][0], miss[i][1]))
        for i in order:
            best_j, best_b = -1, None
            for j in range(M):  # ascending j => lowest column index wins ties
                if not free[j]:
                    continue
                b = float(benefit[i, j])
                if best_j < 0 or b > best_b:
                    best_j, best_b = j, b
            assert best_j >= 0, "greedy_match ran out of free columns"
            match[i] = best_j
            free[best_j] = False
        return match

    # ===================================================================
    # DECODE
    # ===================================================================
    def _plan_decode(
        self, miss: List[ExpertKey], demand: Dict[ExpertKey, int],
        ge: np.ndarray, cache_view: GlobalSlotCacheView, layer_id: int,
    ) -> List[FetchOp]:
        G = self.ep_size
        D = self._build_D(miss, ge)  # [M, G] = D_now
        total_phys = sum(cache_view.per_rank[g].cap for g in range(G))
        miss, D = self._fit_capacity(miss, D, total_phys, layer_id)
        M = len(miss)
        if M == 0:
            return []
        K = A.compute_k(M, G)

        # ---- CAPONLY: mirror PREFILL — cap-only Hungarian, no quota DP ----
        # Precedence: no_placement / greedy win over decode_caponly.
        if self.decode_caponly and not self.no_placement and not self.greedy:
            by_gpu = self._decode_caponly_by_gpu(miss, D, cache_view, K)
            return self._emit_decode(by_gpu, miss, D, demand, cache_view)

        # ---- D1: per-GPU ordered PHYSICAL candidate list ----
        # cand[g] = [(slot_idx, evict_cost, victim_key), ...]; empty first @0,
        # then occupied in LFU order; truncated to qcap_g = min(K, cap).
        cand: List[List[Tuple[int, float, Optional[ExpertKey]]]] = []
        for g in range(G):
            rc = cache_view.per_rank[g]
            lst: List[Tuple[int, float, Optional[ExpertKey]]] = []
            for s in range(rc.cap):
                if rc.slots[s] is None:
                    lst.append((s, 0.0, None))       # empty: cost 0, no victim
            occ = sorted(
                rc.occupied_slots(),
                key=lambda sm: (sm[1].freq, sm[1].last_used, sm[1].inserted_at, sm[0]),
            )
            for slot, meta in occ:
                lst.append((slot, self._evict_cost(meta), meta.expert))
            qcap = min(K, len(lst))
            cand.append(lst[:qcap])

        # ---- D2: prefix cost + quota DP ----
        prefix_cost: List[List[float]] = []
        for g in range(G):
            pc = [0.0]
            for (_s, c, _v) in cand[g]:
                pc.append(pc[-1] + c)
            prefix_cost.append(pc)
        q_g = A.decode_quota_dp(prefix_cost, M)
        assert q_g is not None, (
            f"decode_quota_dp infeasible after capacity fit (M={M}, "
            f"caps={[len(c) for c in cand]})")

        # selected physical slots (flat), each carries its GPU + victim
        selected: List[Tuple[int, int, Optional[ExpertKey]]] = []  # (g, slot, victim)
        for g in range(G):
            for j in range(q_g[g]):
                s, _c, v = cand[g][j]
                selected.append((g, s, v))
        assert len(selected) == M

        # ---- D3: benefit matrix [M, M] + Hungarian ----
        # CurrentLocality(e,g) = D_now(e,g) / sum_g' D_now(e,g')  (algorithm.md §3.1).
        # row sum > 0 (a miss is demanded), so no /0.  float64 — deterministic
        # across ranks (identical inputs); guarded by the determinism test.
        if self.no_placement:
            # ablation: ignore locality — assign misses to the quota-selected slots
            # in fixed order (GPU-grouped by quota). Keeps quota+eviction identical.
            match = np.arange(M)
        elif self.greedy:
            S = D.sum(axis=1, keepdims=True).astype(np.float64)
            current_loc = D.astype(np.float64) / S  # [M, G]
            benefit = self._decode_benefit(miss, current_loc, selected)  # [M, M]
            match = self._greedy_match(miss, benefit)  # stable-home greedy bijection
        else:
            S = D.sum(axis=1, keepdims=True).astype(np.float64)
            current_loc = D.astype(np.float64) / S  # [M, G]
            benefit = self._decode_benefit(miss, current_loc, selected)  # [M, M]
            match = A.decode_match(benefit)  # expert_row -> selected column

        # ---- D4: group by GPU, order by D_now desc, emit + apply ----
        by_gpu: Dict[int, List[Tuple[int, int, Optional[ExpertKey]]]] = {
            g: [] for g in range(G)}
        for i in range(M):
            g, slot, victim = selected[int(match[i])]
            by_gpu[g].append((i, slot, victim))

        return self._emit_decode(by_gpu, miss, D, demand, cache_view)

    def _decode_caponly_by_gpu(
        self, miss: List[ExpertKey], D: np.ndarray,
        cache_view: GlobalSlotCacheView, K: int,
    ) -> Dict[int, List[Tuple[int, int, Optional[ExpertKey]]]]:
        """PREFILL-style placement for decode: cap-only Hungarian over D_now (no
        quota DP).  Hungarian gives expert->GPU (caps=K each); within each GPU pick
        that many slots from the SAME empty-first + LFU-coldest pre-layer candidate
        ordering prefill uses (``_prefill_targets``).  Returns by_gpu = {g:
        [(expert_row, slot, victim), ...]} for the shared emit loop.  Deterministic:
        Hungarian + the candidate sort are fixed-tiebreak -> drift=0."""
        G = self.ep_size
        caps = np.array([min(K, cache_view.per_rank[g].cap) for g in range(G)],
                        dtype=np.int64)
        assign = A.solve_hungarian(D, caps=caps)  # expert_row -> GPU
        rows_by_gpu: Dict[int, List[int]] = {g: [] for g in range(G)}
        for i, g in enumerate(assign):
            rows_by_gpu[int(g)].append(i)
        by_gpu: Dict[int, List[Tuple[int, int, Optional[ExpertKey]]]] = {
            g: [] for g in range(G)}
        for g in range(G):
            rows = rows_by_gpu[g]
            if not rows:
                continue
            # empty-first then LFU-coldest pre-layer victims (same as prefill).
            targets = self._prefill_targets(cache_view.per_rank[g], len(rows))
            for (slot, victim), i in zip(targets, rows):
                by_gpu[g].append((i, slot, victim))
        return by_gpu

    def _emit_decode(
        self, by_gpu: Dict[int, List[Tuple[int, int, Optional[ExpertKey]]]],
        miss: List[ExpertKey], D: np.ndarray, demand: Dict[ExpertKey, int],
        cache_view: GlobalSlotCacheView,
    ) -> List[FetchOp]:
        """Shared decode emit: per GPU order by D_now desc (tie layer,expert),
        contiguous order 0..n-1, emit FetchOp + shadow apply."""
        fetch_ops: List[FetchOp] = []
        for g in range(self.ep_size):
            items = by_gpu[g]
            if not items:
                continue
            # §4: same-GPU fetch order = D_now(e,g) desc, tie (layer, expert)
            items.sort(key=lambda t: (-int(D[t[0], g]), miss[t[0]][0], miss[t[0]][1]))
            rc = cache_view.per_rank[g]
            order = 0
            for (i, slot, victim) in items:
                expert = miss[i]
                fetch_ops.append(FetchOp(
                    expert=expert, fetcher_rank=g, dst_slot=slot,
                    victim_expert=victim, order=order, source="pcie"))
                order += 1
                rc.apply(slot=slot, evict=victim, insert=expert,
                         demand_count=int(demand.get(expert, 0)))
        return fetch_ops

    def _decode_benefit(
        self, miss: List[ExpertKey], current_loc: np.ndarray,
        selected: List[Tuple[int, int, Optional[ExpertKey]]],
    ) -> np.ndarray:
        """[M, len(selected)] placement benefit per (expert, selected slot).

        Stage1 (lambda==0 or no future provider): benefit = CurrentLocality.
        Stage2: + lambda * FutureReuseLocality(e, g) (A2 future affinity,
        normalised; denom 0 -> falls back to CurrentLocality per §3.2).
        """
        gpu_cols = np.array([g for (g, _s, _v) in selected], dtype=np.int64)
        B = current_loc[:, gpu_cols]  # [M, M] float64
        if self.future_lambda > 0.0 and self.future_affinity_provider is not None:
            M, G = current_loc.shape
            fut = np.zeros((M, G), dtype=np.float64)
            for i, (l, e) in enumerate(miss):
                for g in range(G):
                    fut[i, g] = float(self.future_affinity_provider((l, e), g))
            denom = fut.sum(axis=1, keepdims=True)
            future_loc = np.where(denom > 0, fut / np.where(denom > 0, denom, 1.0),
                                  current_loc)
            B = B + self.future_lambda * future_loc[:, gpu_cols]
        return B
