import unittest
from dataclasses import replace

import numpy as np

from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.eviction import DiversityEviction, GateHistory, _rank01, _midrank2
from mgo_v2.cache import GlobalCacheState
from mgo_v2.types import LayerRoutes
from mgo_v2.affinity import AffinityTables
from mgo_v2.admission import AdmissionContext, HungarianAdmission, SwapRefinedAdmission, exact_remote_pairs


class TestController(unittest.TestCase):
    def config(self, **kwargs):
        return replace(RuntimeConfig(num_layers=1, num_experts=4, top_k=2,
                       world_size=1, global_cache_ratio=0.5,
                       eviction="lru", admission="random"), **kwargs)

    def test_exact_mode_does_not_substitute_similar_misses(self):
        ctrl = GlobalExpertController(self.config(substitution_enabled=False), np.ones((1, 4, 4)))
        ctrl.cache.place(0, (0, 0), 0, 0)
        routes = LayerRoutes(0, np.array([0]), np.array([[0, 1]]), np.array([[.9, .1]]))
        plan = ctrl.plan_layer(routes)
        self.assertEqual(plan.substitution.source_to_target, {})
        self.assertEqual(plan.local_exec[0].miss_ops[0][1], 1)

    def test_infeasible_event_preserves_cache(self):
        ctrl = GlobalExpertController(self.config(), np.zeros((1, 4, 4)))
        ctrl.cache.place(0, (0, 0), 0, 0)
        routes = LayerRoutes(0, np.array([0, 0]), np.array([[0, 1], [2, 3]]), np.full((2, 2), .5))
        before = dict(ctrl.cache.owner)
        with self.assertRaisesRegex(RuntimeError, "unpinned slots"):
            ctrl.plan_layer(routes)
        self.assertEqual(ctrl.cache.owner, before)
        self.assertEqual(ctrl.cache.ranks[0].slots, [(0, 0), None])

    def test_affinity_policy_requires_real_inputs(self):
        with self.assertRaisesRegex(ValueError, "requires same_layer"):
            GlobalExpertController(self.config(admission="hungarian_same"), np.eye(4)[None])

    def test_affinity_rejects_wrong_model_shape(self):
        with self.assertRaisesRegex(ValueError, "invalid same_layer"):
            GlobalExpertController(self.config(admission="hungarian_same"), np.eye(4)[None],
                                   AffinityTables(np.ones((1, 3, 3))))

    def test_swap_incremental_objective_matches_full_recomputation(self):
        rng = np.random.default_rng(312)
        for trial in range(30):
            world, n, expert_count = 2 + trial % 3, 5 + trial * 3, 6 + trial % 7
            routes = [{int(e): .25 for e in rng.choice(expert_count, 4, replace=False)} for _ in range(n)]
            preowned = {expert_count - 2: 0, expert_count - 1: world - 1}
            ctx = AdmissionContext(0, list(range(expert_count - 2)), rng.integers(world, size=n),
                                   routes, preowned, GlobalCacheState([expert_count] * world), world)
            seed = HungarianAdmission()
            expected = dict(seed.place(ctx).expert_to_rank)
            current = exact_remote_pairs(ctx, expected)
            experts = sorted(expected)
            for _ in range(2):
                improved = False
                for i, a in enumerate(experts):
                    for b in experts[i + 1:]:
                        ra, rb = expected[a], expected[b]
                        if ra == rb:
                            continue
                        expected[a], expected[b] = rb, ra
                        value = exact_remote_pairs(ctx, expected)
                        if value < current:
                            current, improved = value, True
                        else:
                            expected[a], expected[b] = ra, rb
                if not improved:
                    break
            self.assertEqual(SwapRefinedAdmission(seed).place(ctx).expert_to_rank, expected)

    def test_tied_retention_uses_lru_not_expert_id(self):
        cache = GlobalCacheState([2])
        cache.place(0, (0, 0), 0, 10)
        cache.place(0, (0, 1), 1, 1)
        history = GateHistory(1, 4)
        policy = DiversityEviction(history, np.eye(4)[None])
        self.assertEqual(policy.choose(cache, 0, 0, set()), (0, 1))

    def test_percentile_midranks(self):
        values = {(0, 0): 2, (0, 1): 2, (0, 2): 7}
        self.assertEqual(_rank01(values), {(0, 0): .25, (0, 1): .25, (0, 2): 1.0})

    def test_gate_window_keeps_latest_global_tokens(self):
        history = GateHistory(2, 2, window=3)
        history.update(0, np.array([[1., 0.], [.8, .2]]))
        history.update(1, np.array([[.5, .5]]))
        history.update(0, np.array([[.6, .4], [.3, .7]]))
        self.assertAlmostEqual(history.score(0, 0), (.8 + .6 + .3) / 3)
        self.assertAlmostEqual(history.score(1, 0), .5)

    def test_gate_scores_match_float32_trace_packing(self):
        rng = np.random.default_rng(42)
        observations = rng.random((300, 5), dtype=np.float32)
        history = GateHistory(1, 5, 128)
        for rows in np.array_split(observations, 7):
            history.update(0, rows)
        actual = np.array([history.score(0, e) for e in range(5)], dtype=np.float32)
        expected = observations[-128:].mean(0, dtype=np.float64).astype(np.float32)
        np.testing.assert_array_equal(actual, expected)
        bulk = GateHistory(1, 5, 128)
        bulk.update(0, observations)
        tail = rng.random((17, 5), dtype=np.float32)
        bulk.update(0, tail)
        history.update(0, tail)
        np.testing.assert_array_equal(
            [bulk.score(0, e) for e in range(5)], [history.score(0, e) for e in range(5)])

    def test_all_admission_policies_preserve_quota_and_single_copy(self):
        rng = np.random.default_rng(7)
        affinity = AffinityTables(rng.random((4, 8, 8)), rng.random((3, 8, 8)))
        for policy in ("random", "greedy_current", "greedy_path", "hungarian_current",
                       "hungarian_same", "hungarian_same_path", "hungarian_swap"):
            with self.subTest(policy=policy):
                config = RuntimeConfig(num_layers=4, num_experts=8, top_k=3, world_size=2,
                                       global_cache_ratio=.75, admission=policy, eviction="coverage")
                ctrl = GlobalExpertController(config, np.tile(np.eye(8), (4, 1, 1)), affinity)
                for event in range(20):
                    probs = rng.random((5, 8))
                    probs /= probs.sum(-1, keepdims=True)
                    ids = np.argsort(probs, axis=-1)[:, -3:]
                    weights = np.take_along_axis(probs, ids, axis=-1)
                    weights /= weights.sum(-1, keepdims=True)
                    plan = ctrl.plan_layer(LayerRoutes(event % 4, np.array([0, 0, 0, 1, 1]), ids, weights, probs))
                    counts = np.bincount(list(plan.admission.expert_to_rank.values()), minlength=2).tolist()
                    self.assertEqual(counts, plan.admission.quotas)
                    self.assertLessEqual(max(counts) - min(counts), 1)
                    self.assertEqual(sum(len(p.miss_ops) for p in plan.local_exec.values()), len(plan.substitution.residual_exact_misses))
                    ctrl.cache.assert_consistent()

    def test_combined_percentile_tie_is_exact(self):
        # Reproduces the score tie that diverged at real trace event 37:
        # float normalization gave .9954648526077097 vs .9954648526077098.
        gates = {(0, i): i for i in range(442)}
        damage = {key: 1 for key in gates}
        for i in (437, 439, 441):
            damage[(0, i)] = 0
        for i in (0, 434, 435, 436, 438, 440):
            damage[(0, i)] = 2
        gr, dr = _midrank2(gates), _midrank2(damage)
        self.assertEqual(gr[(0, 437)] + 2 * dr[(0, 437)], gr[(0, 1)] + 2 * dr[(0, 1)])

    def test_directed_global_coverage_with_pinned_neighbor(self):
        sim = np.eye(4)[None]
        sim[0, 2, 0] = sim[0, 2, 1] = .8
        cache = GlobalCacheState([1, 1])
        cache.place(0, (0, 0), 0, 0)
        cache.place(1, (0, 1), 0, 0)
        policy = DiversityEviction(GateHistory(1, 4), sim)
        self.assertEqual(policy.coverage_damage(cache, (0, 0)), 1)
        cache.evict((0, 1))
        self.assertEqual(policy.coverage_damage(cache, (0, 0)), 2)

    def test_incremental_coverage_matches_fresh_global_counts(self):
        rng = np.random.default_rng(135)
        sim = rng.random((3, 8, 8), dtype=np.float32)
        policy = DiversityEviction(GateHistory(3, 8), sim)
        cache = GlobalCacheState([4, 4])
        for event in range(40):
            if event == 20:
                cache = GlobalCacheState([4, 4])
            rank = event % 2
            if cache.ranks[rank].free_slot() is None:
                keys = cache.keys_on_rank(rank)
                cache.evict(keys[int(rng.integers(len(keys)))])
            candidates = [(l, e) for l in range(3) for e in range(8) if (l, e) not in cache.owner]
            key = candidates[int(rng.integers(len(candidates)))]
            cache.place(rank, key, cache.ranks[rank].free_slot(), event)
            for layer, expert in cache.owner:
                residents = sorted(cache.resident_layer(layer))
                counts = policy.neighbors[layer][:, residents].sum(axis=1)
                expected = np.count_nonzero(policy.neighbors[layer, :, expert] & (counts <= 1))
                self.assertEqual(policy.coverage_damage(cache, (layer, expert)), expected)


if __name__ == "__main__":
    unittest.main()
