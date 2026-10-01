"""Independent reference oracle — controller-side cache planning.

Re-implements, from the design spec ONLY, the decisions the production
``GlobalCacheController`` makes per layer:

  Phase 1  global demand union -> hit/miss classify -> hit touch -> owner assign
  Phase 2  per-rank sequential plan (empty-first, else victim) -> fetch_ops

plus a fake physical cache (``RefArcher``) that replays a plan exactly the way
the C++ ``ExpertDispatcher`` is required to: it evicts ONLY the plan's victim,
installs ONLY into the plan's dst_slot, and chooses direct vs staging purely
from whether dst_slot is currently being computed.  It never selects a victim
of its own.

Encoded SPEC (the assumptions this oracle commits to — see SPEC.md):

  * ExpertKey = (layer_id, expert_id).
  * naive owner   : owner_rank(expert) = expert_id % ep_size.
  * balanced owner: count-equal(±1) seeded-random round-robin over shuffled
                    miss_keys (seed = (MOE_EP_RANDOM_SEED, layer_id, layer_seq));
                    demand/id 와 무상관 — 2026-06-10 정의.
  * LRU victim    : argmin (last_used, inserted_at, slot).
  * LFU victim    : argmin (freq,      last_used, inserted_at, slot).
                    freq = ACCESS COUNT (+1 per layer-use, token-agnostic;
                    classic LFU, matches MoE-Infinity entry.visit).
  * hit handling  : a hit refreshes recency and increments freq by 1 (touch)
                    BEFORE Phase-2 victim selection, so a just-used resident is
                    harder to evict.
  * install        : a freshly installed expert gets the newest last_used
                     (touch_counter bumped on apply) and freq = 1 (first use).
  * plan order    : per rank, misses processed by (-demand, layer, expert);
                    empty slot (lowest index) preferred before any eviction.
  * locate         : an expert resident in several ranks resolves to the lowest
                     rank index (matches GlobalSlotCacheView.locate).

NOTHING here imports moe_infinity_ep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

ExpertKey = Tuple[int, int]  # (layer_id, expert_id)


# ===================================================================
# Per-rank slot cache (independent re-impl of RankSlotCache semantics)
# ===================================================================
@dataclass
class RefSlotMeta:
    expert: ExpertKey
    inserted_at: int   # layer_seq at install
    last_used: int     # touch_counter at last use/install
    freq: int          # access count (+1 per use, token-agnostic; classic LFU)


class RefSlotCache:
    """One rank's fixed-size slot array."""

    def __init__(self, rank: int, cap: int):
        self.rank = rank
        self.cap = int(cap)
        self.slots: List[Optional[ExpertKey]] = [None] * self.cap
        self.expert_to_slot: Dict[ExpertKey, int] = {}
        self.meta: List[Optional[RefSlotMeta]] = [None] * self.cap
        self.layer_seq = 0
        self.touch_counter = 0

    # ---- queries ----
    def is_resident(self, key: ExpertKey) -> bool:
        return key in self.expert_to_slot

    def find_slot(self, key: ExpertKey) -> Optional[int]:
        return self.expert_to_slot.get(key)

    def first_empty_slot(self) -> Optional[int]:
        for i, s in enumerate(self.slots):
            if s is None:
                return i
        return None

    def occupied_slots(self) -> List[Tuple[int, RefSlotMeta]]:
        return [(i, m) for i, m in enumerate(self.meta) if m is not None]

    def occupied_keys(self):
        return set(self.expert_to_slot.keys())

    # ---- mutations ----
    def apply(self, slot: int, evict: Optional[ExpertKey],
              insert: Optional[ExpertKey], demand_count: int = 0) -> None:
        if not (0 <= slot < self.cap):
            raise ValueError(f"apply slot {slot} OOB")
        cur = self.slots[slot]
        if evict is None:
            if cur is not None:
                raise AssertionError(f"apply: slot {slot} occupied by {cur}, "
                                     f"expected empty")
        else:
            if cur != evict:
                raise AssertionError(f"apply: slot {slot} holds {cur}, "
                                     f"expected {evict}")
            del self.expert_to_slot[evict]
        if insert is None:
            self.slots[slot] = None
            self.meta[slot] = None
            return
        if insert in self.expert_to_slot:
            raise AssertionError(f"apply: insert {insert} already resident")
        self.slots[slot] = insert
        self.expert_to_slot[insert] = slot
        self.touch_counter += 1
        self.meta[slot] = RefSlotMeta(
            expert=insert, inserted_at=self.layer_seq,
            last_used=self.touch_counter, freq=1)  # access-count LFU: first use

    def touch_use(self, key: ExpertKey, demand_count: int = 0) -> None:
        slot = self.expert_to_slot.get(key)
        if slot is None:
            return
        m = self.meta[slot]
        if m is None:
            return
        self.touch_counter += 1
        m.last_used = self.touch_counter
        m.freq += 1  # access-count LFU: +1 per use (token-agnostic)

    def bump_layer(self) -> None:
        self.layer_seq += 1


# ===================================================================
# Victim policies (independent)
# ===================================================================
def lru_victim(cache: RefSlotCache) -> Optional[int]:
    best, best_score = None, None
    for slot, m in cache.occupied_slots():
        score = (m.last_used, m.inserted_at, slot)
        if best_score is None or score < best_score:
            best, best_score = slot, score
    return best


def lfu_victim(cache: RefSlotCache) -> Optional[int]:
    best, best_score = None, None
    for slot, m in cache.occupied_slots():
        score = (m.freq, m.last_used, m.inserted_at, slot)
        if best_score is None or score < best_score:
            best, best_score = slot, score
    return best


EVICT = {"lru": lru_victim, "lfu": lfu_victim}


def _priority_at(priority, key: ExpertKey) -> float:
    l, e = key
    try:
        return float(priority[l][e])
    except (IndexError, KeyError, TypeError):
        return float("inf")


def priority_victim(cache: RefSlotCache, priority, demand) -> Optional[int]:
    """EAMC: min-priority victim, protecting this-layer-demanded experts.

    Independent re-impl of PriorityEviction: candidates = occupied experts NOT
    in ``demand`` (원본 protected_ondemand); if none, all occupied.  victim =
    argmin (priority[l][e], last_used, inserted_at, slot).  priority None ->
    LRU fallback (matches PriorityEviction's None path).
    """
    if priority is None:
        return lru_victim(cache)
    occ = cache.occupied_slots()
    cands = [(s, m) for s, m in occ if m.expert not in demand] or occ
    best, best_score = None, None
    for slot, m in cands:
        score = (_priority_at(priority, m.expert),
                 m.last_used, m.inserted_at, slot)
        if best_score is None or score < best_score:
            best, best_score = slot, score
    return best


# ===================================================================
# Owner policies (independent)
# ===================================================================
def naive_owner(miss_keys: List[ExpertKey], ep_size: int,
                layer_seq: int = 0) -> Dict[int, List[ExpertKey]]:
    out: Dict[int, List[ExpertKey]] = {r: [] for r in range(ep_size)}
    for (l, e) in miss_keys:
        out[e % ep_size].append((l, e))
    return out


def _stable_seed(*vals: int) -> int:
    # process-independent mix (mirrors owner_policy._stable_seed; builtin hash
    # is per-process salted and must not be used).
    s = 0
    for v in vals:
        s = (s * 1000003 + int(v)) & 0xFFFFFFFF
    return s


def balanced_owner(miss_keys: List[ExpertKey], ep_size: int,
                   layer_seq: int = 0) -> Dict[int, List[ExpertKey]]:
    """count-equal(±1) seeded-random round-robin — mirrors BalancedPlacement
    (2026-06-10 정의: per-rank fetch 개수만 균등, 배정은 demand/id 무관 random)."""
    import os
    import random as _random
    base_seed = int(os.environ.get("MOE_EP_RANDOM_SEED", "0"))
    out: Dict[int, List[ExpertKey]] = {r: [] for r in range(ep_size)}
    keys = sorted(miss_keys)
    if not keys:
        return out
    rng = _random.Random(_stable_seed(base_seed, keys[0][0], layer_seq))
    rng.shuffle(keys)
    off = rng.randrange(ep_size)
    for i, key in enumerate(keys):
        out[(i + off) % ep_size].append(key)
    return out


OWNER = {"naive": naive_owner, "static": naive_owner, "balanced": balanced_owner}


# ===================================================================
# Reference controller — global view + Phase 1/2
# ===================================================================
class RefController:
    def __init__(self, ep_size: int, cap_per_rank: int, num_experts: int,
                 owner: str = "naive", evict: str = "lru"):
        self.ep_size = ep_size
        self.cap = cap_per_rank
        self.num_experts = num_experts
        self.owner_name = owner
        self.evict_name = evict
        self.owner_fn = OWNER[owner]
        self._is_eamc = evict in ("eamc", "priority")
        self.evict_fn = None if self._is_eamc else EVICT[evict]
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
        """Isolated mode: seed slots in slot order (slot0 oldest)."""
        for r, slots in shadow.items():
            rc = self.per_rank[int(r)]
            for i, cell in enumerate(slots):
                if cell is not None:
                    rc.apply(slot=i, evict=None,
                             insert=(int(cell[0]), int(cell[1])), demand_count=0)

    # ---- per-rank sequential plan (mirrors RankPlanner) ----
    def _rank_plan(self, rank: int, miss_experts: List[ExpertKey],
                   demand: Dict[ExpertKey, int], priority=None) -> List[dict]:
        rc = self.per_rank[rank]
        ordered = sorted(miss_experts,
                         key=lambda k: (-int(demand.get(k, 0)), k[0], k[1]))
        ops: List[dict] = []
        order = 0
        for expert in ordered:
            if rc.is_resident(expert):
                continue
            empty = rc.first_empty_slot()
            if empty is not None:
                dst, victim = empty, None
            else:
                if self._is_eamc:
                    vslot = priority_victim(rc, priority, demand)
                else:
                    vslot = self.evict_fn(rc)
                if vslot is None:
                    continue
                dst, victim = vslot, rc.slots[vslot]
            ops.append({"expert": expert, "fetcher_rank": rank, "dst_slot": dst,
                        "victim_expert": victim, "order": order})
            order += 1
            rc.apply(slot=dst, evict=victim, insert=expert,
                     demand_count=int(demand.get(expert, 0)))
        return ops

    # ---- one full layer ----
    def step(self, per_rank_demand: List[List[int]], layer_id: int = 0,
             priority=None) -> dict:
        """per_rank_demand: [ep_size][num_experts] token counts.

        Returns the normalized trajectory for this layer (same shape as the
        production-controller harness produces), then advances layer_seq.
        """
        # Phase 1.1 global demand union
        global_count = [0] * self.num_experts
        for r in range(self.ep_size):
            for e in range(self.num_experts):
                global_count[e] += int(per_rank_demand[r][e])
        demanded = [e for e in range(self.num_experts) if global_count[e] > 0]
        demand: Dict[ExpertKey, int] = {
            (layer_id, e): global_count[e] for e in demanded}

        # Phase 1.2 hit/miss classify (union dedup is implicit: one key per e)
        hit_ops: List[dict] = []
        miss_set: List[ExpertKey] = []
        for e in demanded:
            key = (layer_id, e)
            owner = self.locate(key)
            if owner is None:
                miss_set.append(key)
            else:
                slot = self.per_rank[owner].find_slot(key)
                hit_ops.append({"expert": key, "owner_rank": owner, "slot": slot})

        # Phase 1.3 hit touch (recency/freq refresh BEFORE victim selection)
        for op in hit_ops:
            self.per_rank[op["owner_rank"]].touch_use(
                op["expert"], demand_count=demand.get(op["expert"], 0))

        # Phase 1.4 owner assign (sorted miss set, deterministic; balanced 는
        # (seed, layer_id, layer_seq) 기반 random 이라 layer_seq 를 전달)
        miss_per_rank = self.owner_fn(sorted(miss_set), self.ep_size,
                                      layer_seq=self.per_rank[0].layer_seq)

        # Phase 2 per-rank sequential plan
        fetch_ops: List[dict] = []
        for r in range(self.ep_size):
            fetch_ops.extend(
                self._rank_plan(r, miss_per_rank.get(r, []), demand, priority))

        routing_map_miss = {op["expert"]: op["fetcher_rank"] for op in fetch_ops}
        expert_to_rank = {op["expert"]: op["owner_rank"] for op in hit_ops}
        expert_to_rank.update(routing_map_miss)

        result = {
            "demand": demand,
            "hit_ops": hit_ops,
            "miss_per_rank": miss_per_rank,
            "misses": sorted(miss_set),
            "fetch_ops": fetch_ops,
            "routing_map": expert_to_rank,
            "shadow_after": self.shadow(),
        }
        return result

    def bump(self) -> None:
        for rc in self.per_rank:
            rc.bump_layer()


# ===================================================================
# RefArcher — fake physical cache replaying a plan (Phase 3)
# ===================================================================
class RefArcher:
    """Pure-python stand-in for the C++ ExpertDispatcher's physical cache.

    Models the slot lifecycle (FREE / COMPUTING), direct-vs-staging selection,
    and the controller-victim-only contract.  ``submit_plan`` raises on the
    same conditions the C++ FATALs on: hit-slot mismatch, victim mismatch,
    empty-slot mismatch, PlanQueue order gap.

    Critically: it NEVER chooses a victim.  If a miss op specifies no victim
    but its dst_slot is occupied, that is an error (the controller under-planned).
    """

    FREE, COMPUTING = 0, 1

    def __init__(self, cap: int):
        self.cap = cap
        self.slot_to_key: List[Optional[ExpertKey]] = [None] * cap
        self.slot_state = [self.FREE] * cap
        self.events: List[dict] = []
        self.direct = 0
        self.staging = 0

    def seed_slots(self, slots: List[Optional[ExpertKey]]) -> None:
        self.slot_to_key = [tuple(s) if s is not None else None for s in slots]
        self.slot_state = [self.FREE] * self.cap

    def get_cached_slots(self) -> List[Optional[ExpertKey]]:
        return list(self.slot_to_key)

    def get_cached_experts(self):
        return set(k for k in self.slot_to_key if k is not None)

    def _commit(self, dst_slot, new_key, victim_key, mode):
        held = self.slot_to_key[dst_slot]
        if victim_key is not None:
            if held != victim_key:
                raise AssertionError(
                    f"VICTIM MISMATCH slot={dst_slot} controller_victim="
                    f"{victim_key} archer_holds={held}")
        else:
            if held is not None:
                raise AssertionError(
                    f"EMPTY-SLOT MISMATCH slot={dst_slot} expected empty, "
                    f"archer_holds={held}")
        self.slot_to_key[dst_slot] = new_key
        self.events.append({"event": "commit", "slot": dst_slot,
                            "old": victim_key, "new": new_key, "mode": mode})

    def submit_plan(self, hit_tuples, miss_tuples) -> None:
        """hit_tuples : [(layer, expert, slot), ...]
           miss_tuples: [(layer, expert, dst_slot, victim_layer, victim_expert,
                          order), ...]

        Replays exactly like StartNextFetch: hits validated and marked
        COMPUTING; misses drained in ``order`` with direct/staging chosen by
        dst_slot state.  Staging targets a slot still COMPUTING and is only
        committed after that slot's compute completes (modelled here by
        finishing all hit computes before staging commits).
        """
        self.events.clear()
        # --- hits: validate + mark computing ---
        computing_slots = set()
        for (l, e, slot) in hit_tuples:
            if self.slot_to_key[slot] != (l, e):
                raise AssertionError(
                    f"HIT MISMATCH slot={slot} holds={self.slot_to_key[slot]} "
                    f"controller_hit=({l},{e})")
            self.slot_state[slot] = self.COMPUTING
            computing_slots.add(slot)
            self.events.append({"event": "hit_validate", "slot": slot,
                                "expert": (l, e)})

        # --- misses: order must be contiguous 0..n-1 ---
        ms = sorted(miss_tuples, key=lambda m: m[5])
        for i, m in enumerate(ms):
            if m[5] != i:
                raise AssertionError(
                    f"PlanQueue order gap: got {m[5]} expected {i}")

        # --- drain: direct if dst FREE, else staging (parked) ---
        parked = []
        for (l, e, dst, vl, ve, order) in ms:
            victim_key = (vl, ve) if vl >= 0 and ve >= 0 else None
            if self.slot_state[dst] == self.FREE:
                self.events.append({"event": "fetch_begin", "order": order,
                                    "expert": (l, e), "dst_slot": dst,
                                    "victim": victim_key, "mode": "direct"})
                self._commit(dst, (l, e), victim_key, "direct")
                self.slot_state[dst] = self.COMPUTING
                self.direct += 1
            else:
                self.events.append({"event": "fetch_begin", "order": order,
                                    "expert": (l, e), "dst_slot": dst,
                                    "victim": victim_key, "mode": "staging"})
                self.events.append({"event": "staging_h2d", "dst_slot": dst})
                parked.append((l, e, dst, victim_key))
                self.staging += 1

        # --- compute finishes -> free hit/direct slots -> staging completes ---
        for (l, e, dst, victim_key) in parked:
            # the dst slot's prior compute is now done (wait_victim_compute_done)
            self.events.append({"event": "wait_victim_compute_done",
                                "dst_slot": dst})
            self.slot_state[dst] = self.FREE
            self.events.append({"event": "staging_d2d", "dst_slot": dst})
            self._commit(dst, (l, e), victim_key, "staging")
            self.slot_state[dst] = self.COMPUTING

        # end of layer: all slots free again
        self.slot_state = [self.FREE] * self.cap


# ===================================================================
# small helper to build a per_rank_demand matrix from a sparse dict
# ===================================================================
def demand_matrix(ep_size: int, num_experts: int,
                  per_rank: Dict[int, Dict[int, int]]) -> List[List[int]]:
    m = [[0] * num_experts for _ in range(ep_size)]
    for r, d in per_rank.items():
        for e, c in d.items():
            m[r][e] = c
    return m
