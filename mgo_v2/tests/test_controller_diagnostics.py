import unittest
import numpy as np
from mgo_v2.controller import GlobalExpertController
from mgo_v2.controller_diagnostics import enable_diagnostics, digest
from mgo_v2.config import RuntimeConfig
from mgo_v2.types import LayerRoutes

class DiagnosticParity(unittest.TestCase):
    def test_on_off_full_trajectory(self):
        for policy in ('random', 'hungarian_current'):
            rng = np.random.default_rng(910)
            sim = rng.random((8, 16, 16), dtype=np.float32)
            cfg = RuntimeConfig(num_layers=8, num_experts=16, top_k=4, world_size=4,
                                global_cache_ratio=.75, admission=policy)
            normal, measured = [GlobalExpertController(cfg, sim) for _ in range(2)]
            diagnostic = enable_diagnostics(measured)
            for index in range(160):
                probs = rng.random((1 + index % 17, 16), dtype=np.float32)
                probs /= probs.sum(-1, keepdims=True)
                ids = np.argsort(probs, axis=-1)[:, -4:]
                weights = np.take_along_axis(probs, ids, axis=-1)
                weights /= weights.sum(-1, keepdims=True)
                routes = LayerRoutes(index % 8, np.arange(len(probs)) % 4, ids, weights, probs)
                a, b = normal.plan_layer(routes), measured.plan_layer(routes)
                self.assertEqual(digest(a), digest(b))
                self.assertEqual(normal.cache.owner, measured.cache.owner)
                self.assertEqual([r.slots for r in normal.cache.ranks], [r.slots for r in measured.cache.ranks])
                np.testing.assert_array_equal(normal.history.sums, measured.history.sums)
                self.assertGreaterEqual(diagnostic.last['timer_unattributed_ns'], 0)
            self.assertGreater(sum(r['counts'].get('choose_calls', 0) for r in diagnostic.records), 0)
            self.assertGreater(sum(r['counts'].get('coverage_changed_layers', 0) for r in diagnostic.records), 0)

if __name__ == '__main__':
    unittest.main()
