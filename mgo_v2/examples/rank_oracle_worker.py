#!/usr/bin/env python3
"""Study-only wrapper around the validated benchmark; never changes defaults."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import types

import benchmark_model as bench  # pins one physical GPU before torch import
from mgo_v2.rank_oracle import ExactRankDemandOracle, FrozenRankDemandOracle
from mgo_v2.controller_diagnostics import digest


def raw_digest(routes):
    h = hashlib.sha256(str(routes.layer).encode())
    for x in (routes.origin_ranks, routes.selected_experts, routes.routing_weights, routes.full_router_probs):
        if x is not None:
            h.update(str((x.shape, x.dtype.str)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def next_use_survival(evidence):
    pending = {}
    admitted = observed = survived = 0
    distances = []
    for event, (routes, summary) in enumerate(evidence):
        for expert in set(routes.selected_experts.ravel().tolist()):
            key = (routes.layer, expert)
            if key in pending:
                admission_event, alive = pending.pop(key)
                observed += 1; survived += alive
                distances.append(event-admission_event)
        for rank, hits, misses in summary[5]:
            for layer, expert, slot, victim_layer, victim_expert, order in misses:
                victim = (victim_layer, victim_expert)
                if victim in pending:
                    pending[victim] = (pending[victim][0], False)
                key = (layer, expert)
                if key in pending: raise RuntimeError('unresolved repeated admission')
                pending[key] = (event, True); admitted += 1
    return dict(admissions=admitted, later_raw_demand=observed, survived=survived,
                censored=len(pending), survival_fraction=survived/observed if observed else None,
                next_use_event_distance_sum=sum(distances))


def install_ranges():
    import mgo_v2.runtime as runtime
    state = {'event': -1}
    rank = bench.dist.get_rank()
    original = runtime.DistributedMoERuntime.forward_layer
    def layer(self, *args, **kwargs):
        state['event'] = self.events
        with bench.torch.cuda.nvtx.range(f'oracle:{rank}:{self.events}:moe_layer'):
            return original(self, *args, **kwargs)
    runtime.DistributedMoERuntime.forward_layer = layer
    for name, phase in [('gather_global_routes','routing_metadata'), ('dispatch_tokens','dispatch'), ('return_partials','combine')]:
        old = getattr(runtime, name)
        def call(*args, _old=old, _phase=phase, **kwargs):
            with bench.torch.cuda.nvtx.range(f'oracle:{rank}:{state["event"]}:{_phase}'):
                return _old(*args, **kwargs)
        setattr(runtime, name, call)
    original_execute = bench.LegacySlotExecutorAdapter.execute
    def execute(self, *args, **kwargs):
        with bench.torch.cuda.nvtx.range(f'oracle:{rank}:{state["event"]}:expert_execution'):
            return original_execute(self, *args, **kwargs)
    bench.LegacySlotExecutorAdapter.execute = execute
    # The native submit starts asynchronous H2D and expert execution. The
    # enclosing layer is the association scope for GPU activities on workers.
    class DispatcherProxy:
        def __init__(self, native): self.native = native
        def __getattr__(self, name): return getattr(self.native, name)
        def submit_plan(self, *args):
            with bench.torch.cuda.nvtx.range(f'oracle:{rank}:{state["event"]}:h2d_fetch_submission'):
                return self.native.submit_plan(*args)
    old_init = bench.LegacySlotExecutorAdapter.__init__
    def executor_init(self, dispatcher, *args, **kwargs):
        old_init(self, DispatcherProxy(dispatcher), *args, **kwargs)
    bench.LegacySlotExecutorAdapter.__init__ = executor_init
    return DispatcherProxy


def main():
    if int(os.environ.get('WORLD_SIZE', '0')) != 4 or bench.BOOT['visible_gpu'] != str([0,1,4,5][bench.BOOT['local_rank']]):
        raise RuntimeError('this experiment requires physical GPUs 0,1,4,5 as exactly four ranks')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cells', required=True); parser.add_argument('--output', required=True)
    args, _ = parser.parse_known_args()
    cells = json.loads(Path(args.cells).read_text()); root = Path(args.output)
    # Study metadata lives in cells; benchmark ignores these extra keys.
    current = {}
    old_init = bench.DistributedMoERuntime.__init__
    def runtime_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs); current['runtime'] = self
        if calls >= 1:
            configure(self, cells[calls-1])
    bench.DistributedMoERuntime.__init__ = runtime_init
    original_generate = bench.generate
    calls = 0
    def configure(runtime, cell):
        controller = runtime.controller
        rank = bench.dist.get_rank()
        mode = cell['mode']; policy = cell['policy']
        frozen = None
        if policy == 'O0':
            if mode == 'planning':
                oracle = ExactRankDemandOracle()
                class LeaderOracle:
                    name = oracle.name
                    def place(self, ctx):
                        payload = [None]
                        if rank == 0:
                            try:
                                oracle.place(ctx); payload[0] = dict(record=oracle.records[-1])
                                with (root / (cell['name']+'-solver-progress.jsonl')).open('a') as stream:
                                    stream.write(json.dumps(dict(event=len(oracle.records)-1, **oracle.records[-1]))+'\n')
                            except Exception as exc:
                                payload[0] = dict(error=repr(exc))
                        bench.dist.broadcast_object_list(payload, src=0)
                        if 'error' in payload[0]: raise RuntimeError(payload[0]['error'])
                        if rank != 0: oracle.records.append(payload[0]['record'])
                        return FrozenRankDemandOracle([payload[0]['record']]).place(ctx)
                controller.admission = LeaderOracle()
            else:
                frozen = json.loads(Path(cell['planning_trace'].format(rank=rank)).read_text())
                oracle = FrozenRankDemandOracle(frozen['assignments'])
                controller.admission = oracle
        evidence = []
        original_plan = controller.plan_layer
        def plan(self, routes):
            p = original_plan(routes)
            # Retain evidence, hash/serialize after generation timing. The raw
            # arrays are controller-owned CPU arrays and never mutated later.
            loads = [0]*4; distinct = [0]*4
            for e,r in p.owner_by_expert.items(): distinct[r] += 1
            for token in p.effective_token_routes:
                for e in token: loads[p.owner_by_expert[e]] += 1
            summary = (p.layer, dict(p.substitution.source_to_target),
                       sorted(p.substitution.residual_exact_misses), dict(p.admission.expert_to_rank),
                       p.admission.quotas, [(r, v.hit_ops, v.miss_ops) for r,v in p.local_exec.items()],
                       [tuple(c.slots) for c in self.cache.ranks], loads, distinct)
            evidence.append((routes, summary))
            return p
        controller.plan_layer = types.MethodType(plan, controller)
        current['active'] = (cell, evidence, oracle if policy == 'O0' else None, frozen)

    def generate(*values, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1: return original_generate(*values, **kwargs)
        cell, evidence, oracle, frozen = current['active']
        profiling = cell.get('phase_profile', False)
        if profiling:
            proxy = install_ranges()
            executor = current['runtime'].executor
            executor.dispatcher = proxy(executor.dispatcher)
            bench.torch.cuda.cudart().cudaProfilerStart()
        # benchmark's outer generation timer must exclude validation below.
        result = original_generate(*values, **kwargs)
        if profiling:
            bench.dist.barrier()
            bench.torch.cuda.synchronize()
            bench.torch.cuda.cudart().cudaProfilerStop()
        current['pending'] = (cell, evidence, result[0], oracle, frozen)
        return result
    bench.generate = generate
    # Validation is after the benchmark has stopped its generation timer, at
    # the existing physical-cache check. This also precedes receipt publication.
    original_assert = bench.LegacySlotExecutorAdapter.assert_cache_matches
    def validate(self, cache):
        original_assert(self, cache)
        if 'pending' not in current: return
        cell, evidence, output, oracle, frozen = current.pop('pending')
        current.pop('active', None)
        rank = bench.dist.get_rank()
        rows = [dict(event=i, layer=routes.layer, raw_sha256=raw_digest(routes),
                     plan_cache_sha256=digest(summary[:-2]), loads=summary[-2], distinct=summary[-1])
                for i,(routes,summary) in enumerate(evidence)]
        tokens = output.cpu().tolist()
        receipt = dict(events=rows, tokens=tokens, tokens_sha256=digest(tokens),
                       assignments=oracle.records if oracle else None)
        if frozen is not None:
            if rows != frozen['events'] or tokens != frozen['tokens']:
                raise RuntimeError('frozen route/plan/cache/token parity failed')
            if oracle.cursor != len(oracle.records): raise RuntimeError('unconsumed frozen events')
        receipt['status'] = 'PASS'
        target = root / f'{cell["name"]}-evidence-rank{rank}.json'
        target.write_text(json.dumps(receipt)+'\n')
        if cell['mode'] == 'planning' and rank == 0:
            with (root / f'{cell["name"]}-raw.pkl').open('wb') as stream:
                for event in evidence: pickle.dump(event, stream, protocol=5)
        if rank == 0 and len(evidence) > 96:
            (root / (cell['name']+'-survival.json')).write_text(json.dumps(next_use_survival(evidence))+'\n')
        evidence.clear()
        print(json.dumps(dict(evidence=cell['name'], rank=rank, events=len(rows), parity='PASS')), flush=True)
    bench.LegacySlotExecutorAdapter.assert_cache_matches = validate
    bench.main()

if __name__ == '__main__': main()
