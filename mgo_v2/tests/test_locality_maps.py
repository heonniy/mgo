import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("locality_maps", Path(__file__).parents[1] / "examples/locality_maps.py")
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)


class LocalityMapsTest(unittest.TestCase):
    def test_pair_route_distinction(self):
        origins = np.array([0, 1, 1])
        routes = [[0, 1, 2], [0, 2], []]
        counts = maps.accounting(origins, routes, {0: 0, 1: 0, 2: 1}, 2)
        self.assertEqual(counts["local_token_rank_pairs"], 2)
        self.assertEqual(counts["remote_token_rank_pairs"], 2)
        self.assertEqual(counts["local_expert_routes"], 3)
        self.assertEqual(counts["remote_expert_routes"], 2)
        self.assertEqual(counts["rank_token_loads"], [2, 2])

    def test_fixed_quota_determinism_and_objectives(self):
        rng = np.random.default_rng(7)
        selected = np.stack([rng.choice(20, 4, replace=False) for _ in range(32)])
        origins = np.repeat(np.arange(4), 8)
        first = maps.make_maps(origins, selected, 4, trials=1000)
        self.assertEqual(first, maps.make_maps(origins, selected, 4, trials=1000))
        values = []
        for plan in first:
            self.assertEqual(list(sorted(plan["owners"])), list(range(20)))
            self.assertEqual(np.bincount(list(plan["owners"].values()), minlength=4).tolist(), [5] * 4)
            # Separate set oracle: one pair per unique token/destination.
            pairs = {(t, plan["owners"][int(e)]) for t, row in enumerate(selected) for e in row}
            remote = sum(rank != origins[t] for t, rank in pairs)
            self.assertEqual(remote, plan["accounting"]["remote_token_rank_pairs"])
            values.append(remote)
        self.assertEqual(values, sorted(values))


if __name__ == "__main__":
    unittest.main()
