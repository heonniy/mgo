import itertools
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from mgo_v2.rank_oracle import solve_exact, ExactRankDemandOracle, FrozenRankDemandOracle
from mgo_v2.admission import AdmissionContext, balanced_quotas
from mgo_v2.cache import GlobalCacheState

class OracleTests(unittest.TestCase):
    def test_exhaustive_small(self):
        rng = np.random.default_rng(921)
        for w in (2, 3, 4):
            for n in range(0, 7):
                d = rng.integers(0, 9, n).tolist()
                base = rng.integers(0, 12, w).tolist()
                q = balanced_quotas(n, w)
                candidates = [x for x in itertools.product(range(w), repeat=n)
                              if [x.count(r) for r in range(w)] == q]
                expected = min(candidates, key=lambda x: (max(base[r]+sum(v for v, owner in zip(d,x) if owner == r) for r in range(w)), x))
                ranks, result = solve_exact(d, base, q)
                self.assertEqual(tuple(ranks), expected)
                self.assertEqual(solve_exact(d, base, q)[0], ranks)
                self.assertEqual(result['status'], 'optimal')
    def test_replay_fail_closed(self):
        cache = GlobalCacheState([4]*4)
        cache.place(2, (0, 9), 0, 0)
        ctx = AdmissionContext(0, [1,2,3,4], np.array([0,1]),
                               [{1:.2,9:.8}, {2:.3,3:.3,4:.4}], {9:2}, cache, 4)
        oracle = ExactRankDemandOracle()
        expected = oracle.place(ctx)
        self.assertEqual(cache.owner_of((0,9)), 2)
        replay = FrozenRankDemandOracle(oracle.records)
        self.assertEqual(replay.place(ctx), expected)
        with self.assertRaises(RuntimeError): replay.place(ctx)
        ctx.preowned = {9:3}
        with self.assertRaises(RuntimeError): FrozenRankDemandOracle(oracle.records).place(ctx)
    def test_multiple_tie_blocks(self):
        d=[8,1,2,5,3,8,4,7,1]; base=[4,1]; q=[5,4]
        feasible=[x for x in itertools.product(range(2),repeat=9) if x.count(0)==5]
        expected=min(feasible,key=lambda x:(max(base[r]+sum(v for v,owner in zip(d,x) if owner==r) for r in range(2)),x))
        self.assertEqual(tuple(solve_exact(d,base,q)[0]),expected)
    def test_timeout_has_no_fallback(self):
        with patch('mgo_v2.rank_oracle.milp',return_value=SimpleNamespace(status=1,success=False,message='time limit')):
            with self.assertRaisesRegex(RuntimeError,'exact oracle failed'):
                solve_exact([1,2],[0,0],[1,1])
    def test_controller_pinned_and_capacity(self):
        from mgo_v2.config import RuntimeConfig
        from mgo_v2.controller import GlobalExpertController
        from mgo_v2.types import LayerRoutes
        config=RuntimeConfig(num_layers=1,num_experts=8,top_k=2,world_size=2,
                             global_cache_ratio=.5,admission='random',eviction='lru',substitution_enabled=False)
        ctrl=GlobalExpertController(config,np.eye(8)[None]); ctrl.admission=ExactRankDemandOracle()
        ctrl.cache.place(0,(0,0),0,0); ctrl.cache.place(0,(0,7),1,0)
        ctrl.cache.place(1,(0,6),0,0); ctrl.cache.place(1,(0,5),1,0)
        routes=LayerRoutes(0,np.array([0,1]),np.array([[0,1],[0,2]]),np.full((2,2),.5))
        p=ctrl.plan_layer(routes)
        self.assertEqual(ctrl.cache.owner_of((0,0)),0)
        self.assertEqual(p.admission.quotas,[1,1]); ctrl.cache.assert_consistent()
        before=dict(ctrl.cache.owner)
        impossible=LayerRoutes(0,np.array([0,1,0]),np.array([[0,1],[2,3],[4,5]]),np.full((3,2),.5))
        with self.assertRaisesRegex(RuntimeError,'unpinned slots'): ctrl.plan_layer(impossible)
        self.assertEqual(ctrl.cache.owner,before)
    def test_invalid(self):
        for d,b,q in [([1],[0,0],[0,0]), ([-1],[0],[1]), ([1],[0],[-1])]:
            with self.assertRaises(ValueError): solve_exact(d,b,q)

if __name__ == '__main__': unittest.main()
