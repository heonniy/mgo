import unittest

import numpy as np

from mgo_v2.admission import (
    AdmissionContext,
    HungarianAdmission,
    SwapRefinedAdmission,
    exact_remote_pairs,
)
from mgo_v2.affinity import AffinityTables
from mgo_v2.cache import GlobalCacheState
from mgo_v2.eviction import DiversityEviction, GateHistory, GateScoreEviction
from mgo_v2.substitution import SubstitutionPolicy
from mgo_v2.types import LayerRoutes


class TestSubstitution(unittest.TestCase):
    def test_any_high_route_protects_whole_expert(self):
        sim = np.zeros((1, 4, 4), dtype=np.float32)
        sim[0, 2, 0] = 0.9
        cache = GlobalCacheState([4])
        cache.place(0, (0, 0), 0, 0)
        routes = LayerRoutes(
            0,
            np.array([0, 0]),
            np.array([[2, 1], [2, 3]]),
            np.array([[0.25, 0.10], [0.05, 0.10]], dtype=np.float32),
        )
        result = SubstitutionPolicy(sim).decide(routes, cache)
        self.assertIn(2, result.protected_misses)
        self.assertNotIn(2, result.source_to_target)

    def test_dynamic_resident_fallback(self):
        sim = np.zeros((1, 4, 4), dtype=np.float32)
        sim[0, 2, 1] = 0.75
        cache = GlobalCacheState([4])
        cache.place(0, (0, 1), 0, 0)
        routes = LayerRoutes(
            0,
            np.array([0]),
            np.array([[2, 3]]),
            np.array([[0.10, 0.10]], dtype=np.float32),
        )
        result = SubstitutionPolicy(sim).decide(routes, cache)
        self.assertEqual(result.source_to_target[2], 1)
        self.assertEqual(result.target_tier[2], "inactive_resident")


class TestEviction(unittest.TestCase):
    def test_coverage_protects_unique_region(self):
        sim = np.zeros((1, 3, 3), dtype=np.float32)
        sim[0, 2, 0] = 0.9
        cache = GlobalCacheState([2])
        cache.place(0, (0, 0), 0, 1)
        cache.place(0, (0, 1), 1, 2)

        history = GateHistory(1, 3, 128)
        history.rows[0].append(np.zeros(3))
        history.sums[0] = np.array([0.0, 0.9, 0.0])

        gate = GateScoreEviction(history)
        self.assertEqual(gate.choose(cache, 0, 0, set()), (0, 0))

        coverage = DiversityEviction(history, sim, lam=2.0, k_min=1)
        self.assertEqual(coverage.choose(cache, 0, 0, set()), (0, 1))


class TestAdmission(unittest.TestCase):
    def _ctx(self):
        cache = GlobalCacheState([4, 4])
        routes = [{2: 1.0}, {3: 1.0}]
        return AdmissionContext(
            layer=0,
            incoming=[2, 3],
            origin_ranks=np.array([0, 1]),
            effective_token_routes=routes,
            preowned={},
            cache=cache,
            world_size=2,
            affinity=AffinityTables(),
        )

    def test_hungarian_obeys_hard_quota_and_locality(self):
        ctx = self._ctx()
        result = HungarianAdmission().place(ctx)
        self.assertEqual(result.quotas, [1, 1])
        self.assertEqual(result.expert_to_rank[2], 0)
        self.assertEqual(result.expert_to_rank[3], 1)

    def test_swap_never_increases_exact_remote_pairs(self):
        ctx = self._ctx()
        seed = HungarianAdmission(use_same=True, use_path=True)
        base = seed.place(ctx)
        refined = SwapRefinedAdmission(seed).place(ctx)
        self.assertLessEqual(
            exact_remote_pairs(ctx, refined.expert_to_rank),
            exact_remote_pairs(ctx, base.expert_to_rank),
        )


if __name__ == "__main__":
    unittest.main()
