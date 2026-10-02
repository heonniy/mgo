"""Policy invariants and independent differential checks; CPU only."""
import unittest
import numpy as np
from replica_pareto_cpu import ReplicaReplay, IndependentZeroReplay, ROW_BYTES, traffic


class ReplicaTests(unittest.TestCase):
    def test_greedy_ties_and_incremental_dispatch(self):
        p = ReplicaReplay([4, 4], .5)
        row, ds, _, ops = p.event(0, [0, 1], [[0, 1], [0, 1]], audit_greedy=True)
        replicas = [op for op in ops if op[0] == 'replica']
        self.assertEqual([(o[1], o[2], o[4]) for o in replicas],
                         [(0, 1, ROW_BYTES), (1, 1, 2 * ROW_BYTES)])
        self.assertEqual(row['peer_activation_bytes'], 0)
        self.assertEqual(row['first_copy_fetches'], 2)
        self.assertEqual(row['replica_fetches'], 2)
        np.testing.assert_array_equal(ds, [[0, 0], [1, 1]])

    def test_rank_ties_and_strict_cap(self):
        p = ReplicaReplay([2, 2, 2], 1 / 6)
        row, _, _, ops = p.event(0, [0, 1, 2], [[3], [3], [3]], audit_greedy=True)
        self.assertEqual([(o[1], o[2]) for o in ops], [(3, 0), (3, 1)])
        self.assertEqual(row['duplicate_slots'], 1)
        self.assertEqual(p.primary[(0, 3)], 0)

    def test_active_protection_and_atomic_failure(self):
        p = ReplicaReplay([1, 1], 0)
        p.event(0, [0], [[0]])
        old = p.state()
        with self.assertRaisesRegex(RuntimeError, 'no legal capacity'):
            p.event(0, [0], [[0, 1]])
        self.assertEqual(old, p.state())
        # Rank 1's only resident is active: optional replicas must skip it.
        p = ReplicaReplay([2, 1], 1)
        p.event(0, [1], [[1]])
        row, _, _, _ = p.event(0, [0, 1, 1], [[0], [0], [1]], audit_greedy=True)
        self.assertEqual(row['replica_fetches'], 0)  # rank 1 has no legal victim
        self.assertIn((0, 1), p.ranks[1])

    def test_primary_promotion_and_unique_replica_eviction(self):
        p = ReplicaReplay([1, 1], .5)
        p.event(0, [0, 1], [[0], [0]])
        self.assertEqual(p.primary[(0, 0)], 0)
        row, _, _, _ = p.event(0, [0], [[1]])
        self.assertEqual(p.primary[(0, 0)], 1)
        self.assertEqual(p.duplicates, 0)
        row, _, _, ops = p.event(0, [0, 1], [[1], [1]], audit_greedy=True)
        self.assertNotIn((0, 0), p.owners)  # replica replaces inactive unique expert
        self.assertEqual(row['replica_fetches'], 1)
        row, _, _, _ = p.event(0, [1], [[0]])
        self.assertEqual(row['reload_fetches'], 1)
        self.assertEqual(row['first_copy_fetches'], 0)

    def test_lru_usage_and_key_tie(self):
        p = ReplicaReplay([2, 2], 0)
        p.event(0, [0], [[2, 1]])
        p.event(0, [0], [[2]])
        _, _, _, ops = p.event(0, [0], [[3]])
        self.assertEqual(ops[0][3][1], (0, 1))
        p = ReplicaReplay([2, 2], 0)
        p.event(0, [0], [[2, 1]])
        _, _, _, ops = p.event(0, [0], [[3]])
        self.assertEqual(ops[0][3][1], (0, 1))

    def test_random_zero_independent_and_greedy_exactness(self):
        rng = np.random.default_rng(8741)
        for world in (2, 4):
            capacities = [7] * world
            p = ReplicaReplay(capacities, 0)
            ref = IndependentZeroReplay(capacities)
            replicas = [ReplicaReplay(capacities, rho) for rho in (.125, .25, .5, .75)]
            twins = [ReplicaReplay(capacities, rho) for rho in (.125, .25, .5, .75)]
            for event in range(24):
                origins = np.arange(world * 2) % world
                selected = np.array([rng.choice(6, 2, replace=False) for _ in origins])
                layer = event % 3
                row, ds, t, _ = p.event(layer, origins, selected)
                state, rd, sd, sc, first, reloads = ref.event(layer, origins, selected)
                self.assertEqual(p.state(), state)
                np.testing.assert_array_equal(ds, rd)
                np.testing.assert_array_equal(t['dispatch'], sd)
                np.testing.assert_array_equal(t['combine'], sc)
                self.assertEqual((row['first_copy_fetches'], row['reload_fetches']), (first, reloads))
                self.assertEqual(row['duplicate_slots'], 0)
                for replica, twin in zip(replicas, twins):
                    a = replica.event(layer, origins, selected, audit_greedy=True)
                    b = twin.event(layer, origins, selected)
                    self.assertEqual(a[0], b[0])
                    self.assertEqual(a[3], b[3])
                    self.assertEqual(replica.state(), twin.state())
                    np.testing.assert_array_equal(a[1], b[1])

    def test_hand_counted_peer_rows(self):
        t = traffic([0, 1], [[0, 1, 1], [0, 0, 1]], 2)
        self.assertEqual(t['remote_pairs'], 2)
        self.assertEqual(t['remote_expert_routes'], 4)
        self.assertEqual(t['peer_bytes'], 6 * ROW_BYTES)


if __name__ == '__main__':
    unittest.main()
