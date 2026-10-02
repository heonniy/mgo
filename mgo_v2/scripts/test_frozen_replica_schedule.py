import copy
import unittest
import numpy as np
from replica_pareto_cpu import ReplicaReplay
from frozen_replica_schedule import FrozenReplicaState, compile_event


class FrozenTests(unittest.TestCase):
    def make(self):
        replay = ReplicaReplay([3, 3], .5)
        origin = np.array([0, 1]); selected = np.array([[0, 1], [0, 1]])
        row, dest, tr, ops = replay.event(0, origin, selected)
        return compile_event(0, 0, origin, selected, replay, row, dest, tr, ops, set())

    def test_exact_actions(self):
        event = self.make()
        state = FrozenReplicaState([3, 3], 3)
        state.apply(event)
        self.assertEqual(sum(map(len, state.entries)), 4)

    def test_corruption_rejected(self):
        mutations = [lambda e: e['admissions'][0].update(victim=[9, 9]),
                     lambda e: e['admissions'][0].update(category='reload_fetches'),
                     lambda e: e['destinations'][1].__setitem__(0, 0),
                     lambda e: e['local_exec'][0]['miss_ops'][0].__setitem__(2, 2),
                     lambda e: e.update(post_state_sha256='bad')]
        for change in mutations:
            event = copy.deepcopy(self.make()); change(event)
            with self.assertRaises(AssertionError):
                FrozenReplicaState([3, 3], 3).apply(event)


if __name__ == '__main__': unittest.main()
