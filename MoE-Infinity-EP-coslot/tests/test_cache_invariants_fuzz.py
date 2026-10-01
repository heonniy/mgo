"""Extra correctness situations — randomized cache-invariant fuzz.

Beyond the fixed golden trajectory, this drives the REAL controller through
many random demand steps across several (ep_size, cap) configs and asserts the
cache stays correct at EVERY step:

  * global_unique_demand == hits + misses
  * misses == fetch_ops (post union-dedup, one fetch per miss)
  * no duplicate fetch in a step
  * no expert resident in >1 rank/slot (duplicate_resident == {})
  * a fetch victim really occupies dst_slot in the pre-step physical slots
  * SLOT-LEVEL drift == 0: replaying the controller's own emitted plan onto a
    physical-slot mirror (seeded from the pre-step shadow) reproduces the
    controller's post-step shadow exactly — not just the expert set, the slot
    mapping.

The last check is the strongest: it proves the controller's shadow mutation is
faithfully realizable by an executor that obeys the plan, which is the whole
contract (README acceptance: drift == 0).
"""
from __future__ import annotations

import random

import pytest
import torch

from moe_infinity_ep.controller.global_controller import GlobalCacheController
from moe_infinity_ep.controller.owner_policy import build_owner_policy
from moe_infinity_ep.controller.evict_policy import build_evict_policy
from moe_infinity_ep.controller.rank_planner import RankPlanner


class _SyntheticCollector:
    def __init__(self, ep_size, num_experts):
        self.ep_size = ep_size
        self.num_experts = num_experts
        self.ep_group = None
        self._cur = None

    def load(self, mat):
        self._cur = mat

    def collect_with_count(self, layer_id, mask):
        return self._cur


def _make(ep_size, cap, num_experts, owner, evict):
    return GlobalCacheController(
        ep_size=ep_size, ep_rank=0, cap_per_rank=cap,
        num_layers=1, num_experts=num_experts, ep_group=None,
        owner_policy=build_owner_policy(owner),
        rank_planner=RankPlanner(build_evict_policy(evict)),
        demand_collector=_SyntheticCollector(ep_size, num_experts),
        command_dispatcher=None,
    )


def _slots(ctrl, ep_size):
    return {r: list(ctrl.cache_view.per_rank[r].slots) for r in range(ep_size)}


def _mask_from_counts(counts, num_experts):
    """Build a [T, num_experts] one-hot router mask whose column sums == counts.

    Needed for ep_size==1, where the controller takes the router-mask stub path
    (it ignores demand_collector when ep_size <= 1)."""
    rows = []
    for e in range(num_experts):
        for _ in range(int(counts[e])):
            row = torch.zeros(num_experts, dtype=torch.bool)
            row[e] = True
            rows.append(row)
    if not rows:
        rows.append(torch.zeros(num_experts, dtype=torch.bool))
    return torch.stack(rows, dim=0)


CONFIGS = [
    # (ep_size, cap, num_experts, owner, evict)
    (1, 4, 16, "static", "lru"),
    (2, 4, 16, "static", "lru"),
    (4, 4, 32, "static", "lru"),
    (4, 3, 24, "balanced", "lru"),
    (4, 4, 32, "static", "lfu"),
    (2, 6, 20, "balanced", "lfu"),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: f"ep{c[0]}_cap{c[1]}_{c[3]}_{c[4]}")
def test_invariants_random_steps(cfg):
    ep_size, cap, num_experts, owner, evict = cfg
    ctrl = _make(ep_size, cap, num_experts, owner, evict)
    rng = random.Random(0xC0FFEE ^ hash(cfg))

    for step in range(60):
        # random per-rank demand matrix
        mat = torch.zeros(ep_size, num_experts, dtype=torch.int64)
        for r in range(ep_size):
            k = rng.randint(1, max(1, num_experts // 3))
            for e in rng.sample(range(num_experts), k):
                mat[r, e] = rng.randint(1, 8)
        ctrl.demand_collector.load(mat)
        mask = _mask_from_counts(mat[0], num_experts)  # used only if ep_size==1

        before = _slots(ctrl, ep_size)        # pre-step physical slots
        p1 = ctrl.classify_and_kick_hit(0, mask)
        plan = ctrl.plan_misses(p1, 0)
        after = _slots(ctrl, ep_size)         # controller shadow after mutation

        n_demand = len(p1.demand)
        n_hit = len(p1.hit_launch_info.hit_ops)
        n_miss = sum(len(v) for v in p1.miss_per_rank.values())
        n_fetch = len(plan.fetch_ops)
        tag = f"[{owner}/{evict} ep{ep_size} cap{cap} step{step}]"

        # --- count invariants ---
        assert n_demand == n_hit + n_miss, f"{tag} demand {n_demand} != {n_hit}+{n_miss}"
        assert n_miss == n_fetch, f"{tag} miss {n_miss} != fetch {n_fetch}"

        # --- no duplicate fetch ---
        fetched = [op.expert for op in plan.fetch_ops]
        assert len(fetched) == len(set(fetched)), f"{tag} duplicate fetch"

        # --- duplicate_resident == {} (no expert in >1 rank/slot) ---
        seen = {}
        for r in range(ep_size):
            for s in after[r]:
                if s is not None:
                    seen[s] = seen.get(s, 0) + 1
        dups = {k: v for k, v in seen.items() if v > 1}
        assert not dups, f"{tag} duplicate_resident {dups}"

        # NB: a victim may be an expert inserted EARLIER IN THE SAME STEP when
        # misses > cap, so it is validated sequentially by the replay below
        # (against the running mirror), not against the static pre-step slots.

        # --- SLOT-LEVEL drift == 0: replay plan onto a mirror seeded from
        #     the pre-step slots; must reproduce the controller shadow. ---
        mirror = {r: list(before[r]) for r in range(ep_size)}
        # validate hits point at the right slot
        for op in p1.hit_launch_info.hit_ops:
            assert mirror[op.owner_rank][op.slot] == op.expert, (
                f"{tag} hit {op.expert} not at slot {op.slot}")
        # apply misses per rank in rank-local order
        by_rank = {r: [] for r in range(ep_size)}
        for op in plan.fetch_ops:
            by_rank[op.fetcher_rank].append(op)
        for r in range(ep_size):
            for op in sorted(by_rank[r], key=lambda o: o.order):
                if op.victim_expert is None:
                    assert mirror[r][op.dst_slot] is None, f"{tag} fill into occupied"
                else:
                    assert mirror[r][op.dst_slot] == op.victim_expert
                mirror[r][op.dst_slot] = op.expert
        assert mirror == after, (
            f"{tag} SLOT-LEVEL DRIFT: replay {mirror} != shadow {after}")


def test_lru_is_true_cumulative_recency():
    """Pins the controller's LRU semantics: across steps the victim is the
    expert with the OLDEST last-use (insert or hit), not slot position.

    Regression guard tied to the controller_gold_trajectory step-3 analysis:
    fill {E0,E1,E2,E3} into a cap-4 cache (E0 oldest), hit E1/E2/E3 (refresh),
    then a miss must evict E0 (true LRU), proving recency, not slot index,
    drives eviction.
    """
    ep_size, cap, num_experts = 1, 4, 16
    ctrl = _make(ep_size, cap, num_experts, "static", "lru")

    def step(experts):
        counts = torch.zeros(num_experts, dtype=torch.int64)
        for e in experts:
            counts[e] = 1
        p1 = ctrl.classify_and_kick_hit(0, _mask_from_counts(counts, num_experts))
        return ctrl.plan_misses(p1, 0)

    step([0])            # E0 inserted first (oldest)
    step([1])
    step([2])
    step([3])            # cache full {E0,E1,E2,E3}, E0 least-recently used
    # refresh E1,E2,E3 (NOT E0); then miss E9 must evict the LRU = E0.
    plan = step([1, 2, 3, 9])
    assert len(plan.fetch_ops) == 1
    op = plan.fetch_ops[0]
    assert op.expert == (0, 9)
    assert op.victim_expert == (0, 0), (
        f"true-LRU victim should be E0 (oldest last-use), got {op.victim_expert}")


if __name__ == "__main__":
    for cfg in CONFIGS:
        test_invariants_random_steps(cfg)
    test_lru_is_true_cumulative_recency()
    print("cache invariants fuzz: ALL PASS")
