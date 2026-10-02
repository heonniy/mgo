#!/usr/bin/env python3
"""Full captured-route C0/C1/compact-follower differential gate and CPU diagnostics."""
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
from mgo_v2.affinity import AffinityTables
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.controller_diagnostics import canonical, digest
from mgo_v2.controller_optimized import enable_local_optimization
from mgo_v2.controller_plan_codec import encode_plan, decode_apply
from mgo_v2.controller_components import ComponentDiagnostics
from mgo_v2.rank_oracle import FrozenRankDemandOracle

ROOT=Path('/home/hwlee/mgo-results/controller_overhead_20261002')
ORIGINAL=Path('/home/hwlee/mgo-results/rank_demand_oracle_20261002')
INPUTS=Path('/home/hwlee/mgo-results/runtime_validation_20261001')
OUT=Path(__file__).resolve().parents[1]/'experiments/controller_overhead_20261002'


def cache_state(controller):
    c=controller.cache
    return (c.tick,dict(c.owner),[(tuple(r.slots),{k:(e.slot,e.last_used,e.admitted_at) for k,e in r.entries.items()}) for r in c.ranks])


def stream(path):
    with path.open('rb') as f:
        while True:
            try:yield pickle.load(f)
            except EOFError:return


def main():
    similarity=np.load(INPUTS/'similarity.npy');affinity=AffinityTables.load_npz(str(INPUTS/'affinity.npz'))
    audit=[];summaries=[]
    for policy in ('P0','P1','O0'):
        config=RuntimeConfig(admission='hungarian_current' if policy=='P1' else 'random',same_layer_alpha=.25)
        c0=GlobalExpertController(config,similarity,affinity)
        c1=enable_local_optimization(GlobalExpertController(config,similarity,affinity))
        follower=enable_local_optimization(GlobalExpertController(config,similarity,affinity))
        if policy=='O0':
            records=json.loads((ORIGINAL/'plan_b8/receipts/plan_b8-evidence-rank0.json').read_text())['assignments']
            for c in (c0,c1,follower):c.admission=FrozenRankDemandOracle(records)
        d0,d1=ComponentDiagnostics(c0),ComponentDiagnostics(c1)
        reference=json.loads((ROOT/f'control_C0/receipts/b8_{policy}_C0-evidence-rank0.json').read_text())
        raw=ROOT/f'control_C0/receipts/b8_{policy}_C0-raw.pkl'
        transcript=hashlib.sha256();started=time.time();events=0;apply_ns=0
        for event,(routes,summary) in enumerate(stream(raw)):
            raw_hash=hashlib.sha256(str(routes.layer).encode())
            for array in (routes.origin_ranks,routes.selected_experts,routes.routing_weights,routes.full_router_probs):
                if array is not None:
                    raw_hash.update(str((array.shape,array.dtype.str)).encode());raw_hash.update(array.tobytes())
            assert raw_hash.hexdigest()==reference['events'][event]['raw_sha256'],(policy,event,'captured raw demand')
            p0=d0.plan_layer(routes);p1=d1.plan_layer(routes)
            if canonical(p0)!=canonical(p1):raise AssertionError((policy,event,'full plan mismatch'))
            if cache_state(c0)!=cache_state(c1):raise AssertionError((policy,event,'full cache mismatch'))
            payload=encode_plan(p1,config,c1.cache.tick)
            before=time.perf_counter_ns();p2=decode_apply(payload,follower,routes);apply_ns+=time.perf_counter_ns()-before
            assert canonical(p1)==canonical(p2), (policy,event,'follower plan')
            assert cache_state(c1)==cache_state(follower),(policy,event,'follower cache')
            np.testing.assert_array_equal(c0.history.sums,c1.history.sums)
            np.testing.assert_array_equal(c1.history.sums,follower.history.sums)
            for layer in range(config.num_layers):
                np.testing.assert_array_equal(np.asarray(c0.history.rows[layer]),np.asarray(c1.history.rows[layer]))
                np.testing.assert_array_equal(np.asarray(c1.history.rows[layer]),np.asarray(follower.history.rows[layer]))
            c1.cache.validate_views();c1.eviction.validate_coverage()
            follower.cache.validate_views();follower.eviction.validate_coverage()
            c0.cache.assert_consistent()
            actual=(p0.layer,dict(p0.substitution.source_to_target),sorted(p0.substitution.residual_exact_misses),
                    dict(p0.admission.expert_to_rank),p0.admission.quotas,
                    [(r,v.hit_ops,v.miss_ops) for r,v in p0.local_exec.items()],
                    [tuple(v.slots) for v in c0.cache.ranks])
            expected=reference['events'][event]['plan_cache_sha256']
            assert digest(actual)==expected,(policy,event,'captured physical plan/cache')
            transcript.update(digest((p1,cache_state(c1))).encode());events+=1
        assert events==3120
        for variant,diag in [('C0',d0),('C1',d1)]:
            fields=sorted({k for r in diag.records for k in r['times_ns']})
            counts=sorted({k for r in diag.records for k in r['counts']})
            path=OUT/f'cpu_components_{policy}_{variant}.csv'
            with path.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['event','layer','controller_ns','unattributed_ns']+['ns_'+k for k in fields]+['count_'+k for k in counts],lineterminator='\n')
                writer.writeheader()
                for r in diag.records:
                    writer.writerow({**{k:r[k] for k in ('event','layer','controller_ns','unattributed_ns')},
                        **{'ns_'+k:r['times_ns'].get(k,0) for k in fields},**{'count_'+k:r['counts'].get(k,0) for k in counts}})
            summaries.append(dict(policy=policy,controller=variant,events=events,
                controller_seconds=sum(r['controller_ns'] for r in diag.records)/1e9,
                component_seconds={k:sum(r['times_ns'].get(k,0) for r in diag.records)/1e9 for k in fields},
                counts={k:sum(r['counts'].get(k,0) for r in diag.records) for k in counts}))
        audit.append(dict(policy=policy,events=events,status='PASS',full_plan_cache_history_equal=True,
                          compact_follower_equal=True,captured_physical_plan_equal=True,
                          decision_transcript_sha256=transcript.hexdigest(),follower_apply_cpu_seconds=apply_ns/1e9,
                          validation_wall_seconds=time.time()-started))
        (OUT/'cpu_differential_validation.json').write_text(json.dumps(dict(status='PASS' if len(audit)==3 else 'RUNNING',checks=audit),indent=2)+'\n')
        (OUT/'cpu_component_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(json.dumps(audit[-1]),flush=True)

if __name__=='__main__':main()
