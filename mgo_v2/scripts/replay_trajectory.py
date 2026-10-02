#!/usr/bin/env python3
"""One single-process CPU controller replay worker for one batch size.

Both policies consume each immutable raw route stream, five times, with fresh
cache/history/RNG. Full event accounting on repetition zero; raw exclusive
component times/counts on all five repetitions. No CUDA imports.
"""
import argparse
import bisect
from collections import defaultdict
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
from mgo_v2.config import RuntimeConfig
from mgo_v2.controller import GlobalExpertController
from mgo_v2.controller_diagnostics import enable_diagnostics, _plan, digest

ROOT = Path('/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002')
INPUTS = Path('/home/hwlee/mgo-results/runtime_validation_20261001')


def read_routes(path):
    # Only load trusted traces written by this study's worker.
    out=[]
    with path.open('rb') as f:
        while True:
            try: out.append(pickle.load(f))
            except EOFError: break
    assert len(out)==65*48
    assert all(r.layer==i%48 for i,r in enumerate(out))
    return out


def snapshot(controller, event):
    cache=controller.cache
    eviction=controller.eviction
    residents=[]
    losses={}
    for layer in range(controller.config.num_layers):
        ids=sorted(cache.resident_layer(layer))
        counts=eviction.neighbors[layer][:,ids].sum(axis=1)
        losses[layer]=eviction.neighbors[layer][counts<=eviction.k_min].sum(axis=0)
    for key,rank in sorted(cache.owner.items()):
        entry=cache.ranks[rank].entries[key]
        residents.append(dict(layer=key[0],expert=key[1],rank=rank,slot=entry.slot,
            gate_score=controller.history.score(*key),coverage_damage=int(losses[key[0]][key[1]]),
            admitted_at=entry.admitted_at,last_used=entry.last_used))
    return dict(event=event,residents=residents,cache_sha256=digest(cache.owner))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',type=int,choices=[4,8,16],required=True)
    parser.add_argument('--cpu',type=int,default=None)
    args=parser.parse_args()
    if args.cpu is not None:
        os.sched_setaffinity(0,{args.cpu})
    out=ROOT/'replay'/f'b{args.batch}'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'complete.json').exists()
    similarity=np.load(INPUTS/'similarity.npy')
    summaries=[]
    policies=('random','hungarian_current')
    for source in policies:
        trace=read_routes(ROOT/'physical'/f'b{args.batch}_{source}-routes.pkl')
        demand=defaultdict(list)
        for index, routes in enumerate(trace):
            for expert in np.unique(routes.selected_experts):
                demand[(routes.layer,int(expert))].append(index)
        physical=[json.loads(line) for line in (ROOT/'physical'/f'b{args.batch}_{source}-events-rank0.jsonl').open()]
        baseline={}
        details={}
        selected=set()
        for repeat in range(5):
            for policy in policies:
                config=RuntimeConfig(world_size=4,admission=policy,same_layer_alpha=.25)
                controller=GlobalExpertController(config,similarity)
                stem=f'{source}-to-{policy}-rep{repeat}'
                diag=enable_diagnostics(controller, out/f'{stem}-events.jsonl' if repeat==0 else None)
                # Later repetitions save timers directly; avoid unneeded trace hashing/serialization.
                diag.records=None
                totals=defaultdict(int)
                counts=defaultdict(int)
                total_ns=0
                admissions=evictions=0
                snapshots=[]
                started=time.time()
                with (out/f'{stem}-timings.jsonl').open('w') as timed:
                    for index,routes in enumerate(trace):
                        if repeat==0:
                            plan=controller.plan_layer(routes)
                            event=diag.last
                            if policy==source:
                                assert event['plan_sha256']==physical[index]['plan_sha256'], (index,'own plan mismatch')
                                assert event['cache_sha256']==physical[index]['cache_sha256'], (index,'own cache mismatch')
                            elapsed=event['controller_ns']
                        else:
                            diag.reset()
                            began=time.perf_counter_ns()
                            plan=_plan(controller,routes)
                            elapsed=time.perf_counter_ns()-began
                        total_ns+=elapsed
                        for key,value in diag.times.items(): totals[key]+=value
                        for key,value in diag.counts.items(): counts[key]+=value
                        admissions+=sum(len(p.miss_ops) for p in plan.local_exec.values())
                        evictions+=sum(op[3]>=0 for p in plan.local_exec.values() for op in p.miss_ops)
                        timed.write(json.dumps(dict(event=index,controller_ns=elapsed,times_ns=dict(diag.times),
                            calls=dict(diag.calls),counts=dict(diag.counts)),separators=(',',':'))+'\n')
                        if repeat==1 and index in selected:
                            state=snapshot(controller,index)
                            state['execution_owners']=plan.owner_by_expert
                            for resident in state['residents']:
                                occurrences=demand[(resident['layer'],resident['expert'])]
                                position=bisect.bisect_right(occurrences,index)
                                resident['next_raw_demand_distance']=occurrences[position]-index if position<len(occurrences) else None
                            snapshots.append(state)
                final=dict(cache=digest(controller.cache.owner),plan=digest(plan),history=digest(controller.history.sums),
                           admissions=admissions,evictions=evictions)
                if repeat==0: baseline[policy]=final
                else: assert final==baseline[policy], 'Replay repetitions changed state or decisions'
                diag.close()
                if snapshots:
                    (out/f'{source}-to-{policy}-snapshots.json').write_text(json.dumps(snapshots))
                row=dict(batch=args.batch,source_policy=source,policy=policy,repeat=repeat,
                         events=len(trace),controller_ns=total_ns,times_ns=dict(totals),counts=dict(counts),
                         final=final,wall_seconds=time.time()-started,cpu_affinity=sorted(os.sched_getaffinity(0)),
                         own_trajectory_verified=policy==source)
                summaries.append(row)
                (out/f'{stem}-summary.json').write_text(json.dumps(row,indent=2)+'\n')
                print(json.dumps(dict(batch=args.batch,source=source,policy=policy,repeat=repeat,
                    controller_seconds=total_ns/1e9,complete=True)),flush=True)
                if repeat==0:
                    details[policy]=[json.loads(line) for line in (out/f'{stem}-events.jsonl').open()]
                del controller,diag
                gc.collect()
            if repeat==0:
                deltas=[(len(c['admissions'])-len(r['admissions']),i) for i,(r,c) in enumerate(zip(details['random'],details['hungarian_current'])) if i>=48]
                # Prespecified selection: largest positive/negative instantaneous fetch divergence,
                # retaining earlier-event ties. Same event indexes for both policies.
                selected={i for _,i in sorted(deltas,key=lambda x:(x[0],x[1]))[:5]}
                selected.update(i for _,i in sorted(deltas,key=lambda x:(-x[0],x[1]))[:5])
                (out/f'{source}-selected-events.json').write_text(json.dumps(sorted(selected)))
    (out/'complete.json').write_text(json.dumps(dict(status='PASS',batch=args.batch,replays=len(summaries),summaries=summaries),indent=2)+'\n')

if __name__=='__main__':
    main()
