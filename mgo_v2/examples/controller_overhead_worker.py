#!/usr/bin/env python3
"""Separate controller-stage wrapper; frozen benchmark and primary packet intact."""
import argparse
import json
import os
import pickle
import types
from pathlib import Path

import benchmark_model as bench  # pin the physical GPU before torch import
from rank_oracle_worker import raw_digest
from mgo_v2.controller_diagnostics import digest
from mgo_v2.controller_optimized import enable_local_optimization
from mgo_v2.rank_oracle import FrozenRankDemandOracle
from mgo_v2.single_planner import SinglePlanner

ORIGINAL = Path('/home/hwlee/mgo-results/rank_demand_oracle_20261002')


def install_ranges(current):
    import mgo_v2.runtime as runtime
    rank = bench.dist.get_rank()
    def label(phase):
        return f'controller:{current["cell"]["name"]}:{rank}:{current["event"]}:{phase}'
    old = runtime.DistributedMoERuntime.forward_layer
    def layer(self, *args, **kwargs):
        current['event'] = self.events
        with bench.torch.cuda.nvtx.range(label('moe_layer')):
            return old(self, *args, **kwargs)
    runtime.DistributedMoERuntime.forward_layer = layer
    for name, phase in [('gather_global_routes','routing_metadata'),('dispatch_tokens','dispatch'),('return_partials','combine')]:
        original = getattr(runtime, name)
        def call(*args, _original=original, _phase=phase, **kwargs):
            with bench.torch.cuda.nvtx.range(label(_phase)):
                return _original(*args, **kwargs)
        setattr(runtime, name, call)
    execute = bench.LegacySlotExecutorAdapter.execute
    def expert(self, *args, **kwargs):
        with bench.torch.cuda.nvtx.range(label('expert_execution')):
            return execute(self, *args, **kwargs)
    bench.LegacySlotExecutorAdapter.execute = expert


class TransportDiagnostics:
    def __init__(self, current):
        self.current = current
        self.records = []
        self.times = {}
    def span(self, label):
        from contextlib import contextmanager
        import time
        @contextmanager
        def scope():
            start = time.perf_counter_ns()
            with bench.torch.cuda.nvtx.range(f'controller:{self.current["cell"]["name"]}:{bench.dist.get_rank()}:{self.current["event"]}:{label}'):
                try: yield
                finally:
                    self.times[label] = self.times.get(label,0) + time.perf_counter_ns()-start
        return scope()


def main():
    local_rank=bench.BOOT['local_rank']
    if int(os.environ.get('WORLD_SIZE','0'))!=4 or local_rank not in range(4) or bench.BOOT['visible_gpu']!=str([0,1,4,5][local_rank]):
        raise RuntimeError('controller stage requires exactly physical GPUs 0,1,4,5')
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cells',required=True);parser.add_argument('--output',required=True)
    args,_=parser.parse_known_args()
    cells=json.loads(Path(args.cells).read_text());root=Path(args.output)
    if bench.dist.is_initialized(): raise RuntimeError('unexpected early distributed initialization')
    current={};calls=0;ranges_installed=False
    original_init=bench.DistributedMoERuntime.__init__
    def runtime_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        current['runtime']=self
        if calls>=1: configure(self,cells[calls-1])
    bench.DistributedMoERuntime.__init__=runtime_init

    def configure(runtime,cell):
        c=runtime.controller;rank=bench.dist.get_rank()
        if bench.dist.get_world_size()!=4 or bench.BOOT['visible_gpu']!=str([0,1,4,5][rank]):
            raise RuntimeError('controller stage requires exactly physical GPUs 0,1,4,5')
        current['cell']=cell;current['event']=0
        variant,policy=cell['controller'],cell['policy']
        baseline=json.loads((ORIGINAL/f'timing_b8/receipts/b8_{policy}_rep0-evidence-rank{rank}.json').read_text())
        oracle=None
        if policy=='O0':
            trace=json.loads((ORIGINAL/f'plan_b8/receipts/plan_b8-evidence-rank{rank}.json').read_text())
            oracle=FrozenRankDemandOracle(trace['assignments']);c.admission=oracle
        if variant in ('C1','C2'): enable_local_optimization(c)
        diagnostic=TransportDiagnostics(current) if cell.get('phase_profile') else None
        planner=SinglePlanner(c,bench.torch,bench.dist,diagnostic) if variant=='C2' else None
        evidence=[];original_plan=c.plan_layer
        def plan(self,routes):
            if diagnostic: diagnostic.times={}
            p=original_plan(routes)
            loads=[0]*4;distinct=[0]*4
            for expert,rank_ in p.owner_by_expert.items():distinct[rank_]+=1
            for token in p.effective_token_routes:
                for expert in token:loads[p.owner_by_expert[expert]]+=1
            summary=(p.layer,dict(p.substitution.source_to_target),sorted(p.substitution.residual_exact_misses),
                     dict(p.admission.expert_to_rank),p.admission.quotas,
                     [(r,v.hit_ops,v.miss_ops) for r,v in p.local_exec.items()],
                     [tuple(v.slots) for v in self.cache.ranks],loads,distinct)
            evidence.append((routes,summary))
            if diagnostic: diagnostic.records.append(dict(event=len(evidence)-1,times_ns=dict(diagnostic.times)))
            return p
        c.plan_layer=types.MethodType(plan,c)
        current['active']=(cell,evidence,baseline,oracle,planner,diagnostic)

    original_generate=bench.generate
    def generate(*args,**kwargs):
        nonlocal calls,ranges_installed
        calls+=1
        if calls==1:return original_generate(*args,**kwargs)
        cell,*_=current['active']
        if cell.get('phase_profile') and not ranges_installed:
            install_ranges(current);ranges_installed=True
            bench.torch.cuda.cudart().cudaProfilerStart()
        result=original_generate(*args,**kwargs)
        if cell.get('phase_profile') and calls==len(cells)+1:
            bench.dist.barrier();bench.torch.cuda.synchronize()
            bench.torch.cuda.cudart().cudaProfilerStop()
        current['pending']=(*current['active'],result[0])
        return result
    bench.generate=generate
    original_assert=bench.LegacySlotExecutorAdapter.assert_cache_matches
    def validate(self,cache):
        original_assert(self,cache)
        if 'pending' not in current:return
        cell,evidence,baseline,oracle,planner,diagnostic,output=current.pop('pending')
        current.pop('active',None)
        rank=bench.dist.get_rank();tokens=output.cpu().tolist()
        rows=[dict(event=i,layer=routes.layer,raw_sha256=raw_digest(routes),
                   plan_cache_sha256=digest(summary[:-2]),loads=summary[-2],distinct=summary[-1])
              for i,(routes,summary) in enumerate(evidence)]
        assert rows==baseline['events'][:len(rows)],'original event/route/plan/cache parity'
        assert tokens==[x[:len(tokens[0])] for x in baseline['tokens']],'original full token parity'
        assert len(rows)==48*len(tokens[0])
        if oracle:assert oracle.cursor==len(rows)
        controller=current['runtime'].controller
        if cell['controller']!='C0':
            controller.cache.validate_views();controller.eviction.validate_coverage()
        result=dict(status='PASS',controller=cell['controller'],policy=cell['policy'],events=rows,
                    tokens=tokens,tokens_sha256=digest(tokens),original_packet_parity=True)
        (root/f'{cell["name"]}-evidence-rank{rank}.json').write_text(json.dumps(result)+'\n')
        meta=dict(status='PASS',controller=cell['controller'],policy=cell['policy'],events=len(rows),
                  planner_rank=0 if planner else None,payload_bytes_per_event=planner.payload_bytes_per_event if planner else 0,
                  logical_payload_bytes=planner.events*planner.payload_bytes_per_event if planner else 0,
                  modeled_planner_peer_tx_bytes=planner.events*planner.payload_bytes_per_event*3 if planner and rank==0 else 0,
                  transport_diagnostics=diagnostic.records if diagnostic else None)
        (root/f'{cell["name"]}-controller-rank{rank}.json').write_text(json.dumps(meta)+'\n')
        if cell['controller']=='C0' and rank==0:
            with (root/f'{cell["name"]}-raw.pkl').open('wb') as f:
                for e in evidence:pickle.dump(e,f,protocol=5)
        evidence.clear()
        print(json.dumps(dict(cell=cell['name'],rank=rank,parity='PASS')),flush=True)
    bench.LegacySlotExecutorAdapter.assert_cache_matches=validate
    bench.main()

if __name__=='__main__':main()
