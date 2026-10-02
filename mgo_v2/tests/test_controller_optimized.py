import copy
import unittest
import numpy as np

from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.controller_diagnostics import canonical
from mgo_v2.controller_optimized import enable_local_optimization, midrank2_array, IndexedHistory
from mgo_v2.controller_plan_codec import encode_plan, decode_apply
from mgo_v2.eviction import _midrank2
from mgo_v2.types import LayerRoutes


def cache_state(c):
    return (dict(c.cache.owner), c.cache.tick,
            [(list(r.slots), {k:(e.slot,e.last_used,e.admitted_at) for k,e in r.entries.items()})
             for r in c.cache.ranks])


class OptimizedControllerTests(unittest.TestCase):
    def test_exact_midrank_ties(self):
        rng = np.random.default_rng(16)
        for n in (0,1,2,17,451):
            for values in (rng.integers(0,8,n), rng.random(n), np.ones(n)):
                expected = _midrank2({(0,i):float(x) for i,x in enumerate(values)})
                np.testing.assert_array_equal(midrank2_array(values), [expected[0,i] for i in range(n)])

    def test_history_float32_rounding(self):
        h = IndexedHistory(3,16,13); rng = np.random.default_rng(23)
        for i in range(80):
            layer = i%3
            h.update(layer, rng.random((i%19+1,16), dtype=np.float32))
            np.testing.assert_array_equal(h.scores, [[h.score(l,e) for e in range(16)] for l in range(3)])

    def test_full_differential_and_codec(self):
        for world in (2,4):
            for policy in ('random','hungarian_current'):
                for lam in (2.0,1.3):
                    rng=np.random.default_rng(427)
                    sim=rng.choice(np.array([.1,.65,.8,1.],dtype=np.float32),size=(8,16,16))
                    for a in sim: np.fill_diagonal(a,1.)
                    config=RuntimeConfig(num_layers=8,num_experts=16,top_k=2,world_size=world,
                                         global_cache_ratio=.5,admission=policy,coverage_lambda=lam,gate_window=13)
                    c0=GlobalExpertController(config,sim)
                    c1=enable_local_optimization(GlobalExpertController(config,sim),debug=True)
                    follower=enable_local_optimization(GlobalExpertController(config,sim),debug=True)
                    victims=0
                    for event in range(100):
                        selected=np.array([rng.choice(16,2,replace=False) for _ in range(4)])
                        weights=rng.random((4,2),dtype=np.float32);weights/=weights.sum(axis=1)[:,None]
                        probs=rng.random((4,16),dtype=np.float32);probs/=probs.sum(axis=1)[:,None]
                        routes=LayerRoutes(event%8,np.arange(4)%world,selected,weights,probs)
                        p0=c0.plan_layer(routes);p1=c1.plan_layer(routes)
                        self.assertEqual(canonical(p0),canonical(p1))
                        self.assertEqual(cache_state(c0),cache_state(c1))
                        payload=encode_plan(p1,config,c1.cache.tick)
                        p2=decode_apply(payload,follower,routes)
                        self.assertEqual(canonical(p1),canonical(p2))
                        self.assertEqual(cache_state(c1),cache_state(follower))
                        np.testing.assert_array_equal(c0.history.sums,c1.history.sums)
                        np.testing.assert_array_equal(c1.history.sums,follower.history.sums)
                        c1.cache.validate_views();c1.eviction.validate_coverage()
                        follower.cache.validate_views();follower.eviction.validate_coverage()
                        victims+=sum(op[3]>=0 for local in p1.local_exec.values() for op in local.miss_ops)
                    self.assertGreater(victims,0)

    def test_frozen_oracle_codec_cursor_and_context(self):
        from mgo_v2.rank_oracle import ExactRankDemandOracle, FrozenRankDemandOracle
        config=RuntimeConfig(num_layers=2,num_experts=8,top_k=2,world_size=2,
                             global_cache_ratio=.75,admission='random',substitution_enabled=False)
        sim=np.repeat(np.eye(8,dtype=np.float32)[None],2,axis=0)
        base=GlobalExpertController(config,sim);base.admission=ExactRankDemandOracle()
        local=enable_local_optimization(GlobalExpertController(config,sim))
        follower=enable_local_optimization(GlobalExpertController(config,sim))
        local.admission=FrozenRankDemandOracle(base.admission.records)
        follower.admission=FrozenRankDemandOracle(base.admission.records)
        rng=np.random.default_rng(19)
        for event in range(12):
            chosen=np.array([rng.choice(8,2,replace=False) for _ in range(2)])
            routes=LayerRoutes(event%2,np.arange(2),chosen,np.full((2,2),.5),np.full((2,8),.125))
            p0=base.plan_layer(routes);p1=local.plan_layer(routes)
            p2=decode_apply(encode_plan(p1,config,local.cache.tick),follower,routes)
            self.assertEqual(canonical(p0),canonical(p1));self.assertEqual(canonical(p1),canonical(p2))
            self.assertEqual(cache_state(base),cache_state(local));self.assertEqual(cache_state(local),cache_state(follower))
        self.assertEqual(local.admission.cursor,12);self.assertEqual(follower.admission.cursor,12)

    def test_component_diagnostics_preserve_decisions(self):
        from mgo_v2.controller_components import ComponentDiagnostics
        config=RuntimeConfig(num_layers=4,num_experts=8,top_k=2,world_size=2,
                             global_cache_ratio=.5,admission='hungarian_current',substitution_enabled=False)
        sim=np.repeat(np.eye(8,dtype=np.float32)[None],4,axis=0)
        bare=GlobalExpertController(config,sim)
        c0=GlobalExpertController(config,sim);d0=ComponentDiagnostics(c0)
        c1=enable_local_optimization(GlobalExpertController(config,sim));d1=ComponentDiagnostics(c1)
        rng=np.random.default_rng(33)
        for event in range(40):
            selected=np.array([rng.choice(8,2,replace=False) for _ in range(2)])
            routes=LayerRoutes(event%4,np.arange(2),selected,np.full((2,2),.5),np.full((2,8),.125))
            expected=bare.plan_layer(routes)
            self.assertEqual(canonical(expected),canonical(d0.plan_layer(routes)))
            self.assertEqual(canonical(expected),canonical(d1.plan_layer(routes)))
            self.assertEqual(cache_state(bare),cache_state(c0));self.assertEqual(cache_state(bare),cache_state(c1))
        for diag in (d0,d1):
            self.assertEqual(len(diag.records),40)
            self.assertTrue(all(r['unattributed_ns']>=0 for r in diag.records))
        self.assertEqual(sum(r['counts'].get('candidate_visits',0) for r in d0.records),
                         sum(r['counts'].get('candidate_visits',0) for r in d1.records))

    def test_payload_failure_is_atomic(self):
        config=RuntimeConfig(num_layers=2,num_experts=8,top_k=2,world_size=2,
                             global_cache_ratio=1.,admission='random')
        sim=np.repeat(np.eye(8,dtype=np.float32)[None],2,axis=0)
        leader=enable_local_optimization(GlobalExpertController(config,sim))
        routes=LayerRoutes(0,np.array([0,1]),np.array([[0,1],[2,3]]),np.full((2,2),.5))
        plan=leader.plan_layer(routes);good=encode_plan(plan,config,leader.cache.tick)
        variants=[]
        for index,value in [(0,0),(2,-1),(3,1),(4,9),(5,99),(8,100),(10+2,16),(10+3,4),(10+5,100)]:
            bad=good.copy();bad[index]=value;variants.append(bad)
        variants.extend([good[:-1],good.astype(np.int64)])
        for bad in variants:
            follower=enable_local_optimization(GlobalExpertController(config,sim))
            before=cache_state(follower)
            with self.assertRaises((ValueError,RuntimeError)):
                decode_apply(bad,follower,routes)
            self.assertEqual(cache_state(follower),before)
            self.assertFalse(any(follower.history.rows))

    def test_manual_mutations_and_reinstall_rejection(self):
        config=RuntimeConfig(num_layers=2,num_experts=8,top_k=2,world_size=2,admission='random',global_cache_ratio=1.)
        c=enable_local_optimization(GlobalExpertController(config,np.repeat(np.eye(8)[None],2,axis=0)))
        c.cache.place(0,(0,1),3,7);c.cache.place(1,(1,2),0,8)
        c.cache.touch((0,1),9);c.cache.evict((1,2));c.cache.place(1,(0,7),2,10)
        c.cache.validate_views();c.eviction.validate_coverage()
        view=c.cache.resident_layer(0)
        self.assertIs(view,c.cache.resident_layer(0))
        self.assertIsInstance(view,frozenset)
        with self.assertRaises(ValueError):enable_local_optimization(c)

if __name__=='__main__':unittest.main()
