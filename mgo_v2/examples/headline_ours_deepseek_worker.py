"""DeepSeek-V2-Lite main_OURS: live Near, bounded cache, pinned source, native experts.

The checkpoint's routed expert weights stay in rank-private pinned host memory.
All dense, attention and shared-expert weights are resident on each GPU.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import time
import types
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT = pin_rank_before_cuda_import()

import numpy as np
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from env_offload_layout import plan_rank_partial_layout
from env_offload_policy import Policy
from env_offload_tensors import pack_rank_partial_layout
from mgo_v2.coalesced_return import combine_rank_partials
from mgo_v2.deepseek_native_expert import DeepseekNativeExpertExecutor
from mgo_v2.eviction import GateHistory
from mgo_v2.fused_transport import FusedTokenRankTransport
from mgo_v2.live_metadata import LiveMetadata
from mgo_v2.pinned_h2d import PriorityH2DScheduler
from mgo_v2.runtime import gather_global_routes

ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
MODEL = Path('/home/hwlee/model/DeepSeek-V2-Lite-Chat')
LAYERS, EXPERTS, TOPK, HIDDEN, EXPERT_ELEMENTS = 26, 64, 6, 2048, 8650752
EXPERT = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
PARTS = {'gate_proj': (0, 2883584), 'up_proj': (2883584, 2883584),
         'down_proj': (5767168, 2883584)}
DIAG = os.environ.get('MGO_DEEPSEEK_CACHE_DIAG') == '1'


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def load_model_and_store():
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    config._attn_implementation = 'sdpa'
    assert (config.num_hidden_layers, config.n_routed_experts,
            config.num_experts_per_tok) == (27, EXPERTS, TOPK)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    weights = json.loads((MODEL / 'model.safetensors.index.json').read_text())['weight_map']
    expected = set(dict(model.named_parameters()))
    assert set(weights) == expected, 'checkpoint/model key mismatch'
    pool = torch.empty((LAYERS * EXPERTS, EXPERT_ELEMENTS),
                       dtype=torch.bfloat16, pin_memory=True)
    assert pool.is_pinned()
    loaded = np.zeros((LAYERS * EXPERTS, 3), dtype=np.bool_)
    for shard in sorted(set(weights.values())):
        with safe_open(str(MODEL / shard), framework='pt', device='cpu') as file:
            for name in sorted(n for n, location in weights.items() if location == shard):
                tensor = file.get_tensor(name)
                match = EXPERT.fullmatch(name)
                if match:
                    layer, expert = int(match[1]) - 1, int(match[2])
                    assert 0 <= layer < LAYERS and 0 <= expert < EXPERTS
                    key = layer * EXPERTS + expert
                    part = match[3]
                    offset, size = PARTS[part]
                    assert tensor.numel() == size
                    pool[key, offset:offset + size].copy_(tensor.reshape(-1))
                    loaded[key, list(PARTS).index(part)] = True
                else:
                    set_module_tensor_to_device(model, name, 'cuda:0', value=tensor,
                                                dtype=torch.bfloat16)
    assert loaded.all(), 'incomplete routed expert store'
    assert all(p.is_cuda for n, p in model.named_parameters() if not EXPERT.fullmatch(n))
    model.generation_config = GenerationConfig.from_pretrained(MODEL, local_files_only=True)
    model.eval()
    return model, pool, [(pool[key],) for key in range(LAYERS * EXPERTS)]


class DeepSeekRuntime:
    def __init__(self, batch, capacities, sources, pool):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.batch = batch
        self.capacities = capacities
        self.cap = capacities[self.rank]
        self.sources = sources
        self.pool = pool
        self.cache = torch.empty((self.cap + 2, EXPERT_ELEMENTS), dtype=torch.bfloat16,
                                 device='cuda')
        self.native_executor = DeepseekNativeExpertExecutor()
        self.reset()

    def reset(self):
        if hasattr(self, 'h2d'):
            self.h2d.close()
        self.index = 0
        self.keys = np.full(self.cap + 2, -1, np.int32)
        self.policy = Policy(self.capacities, np.zeros((LAYERS, EXPERTS, EXPERTS),
                                                    np.float32), False, 7, 42)
        self.history = GateHistory(LAYERS, EXPERTS, 128)
        self.metadata = LiveMetadata(self.batch, self.history, TOPK, EXPERTS, LAYERS)
        self.transport = FusedTokenRankTransport('rank-partial', topk=TOPK)
        self.h2d = PriorityH2DScheduler(self.cache, direct_pinned=True)
        self.ready_metrics = {'waits': 0, 'ready_before_first_wait': 0}
        self.diag = DIAG
        if self.diag:
            self.diag_spans = {key: [] for key in ('dispatch', 'expert', 'return', 'h2d_wait')}
            self.diag_route_host_s = 0.0
            self.diag_h2d_host_wait_s = 0.0
            original_wait_for_slot = self.h2d.wait_for_slot

            def timed_wait_for_slot(slot):
                if self.index < LAYERS:
                    return original_wait_for_slot(slot)
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
                started = time.perf_counter()
                original_wait_for_slot(slot)
                self.diag_h2d_host_wait_s += time.perf_counter() - started
                after.record()
                self.diag_spans['h2d_wait'].append((before, after))

            self.h2d.wait_for_slot = timed_wait_for_slot

    def install(self, model):
        for layer, block in enumerate(model.model.layers[1:]):
            block.mlp.moe = types.MethodType(self._moe(layer), block.mlp)

    def _moe(self, layer):
        rt = self

        def forward(module, hidden, selected, weights):
            assert hidden.shape == (rt.batch if rt.index >= LAYERS else hidden.shape[0], HIDDEN)
            assert selected.shape == weights.shape == (hidden.shape[0], TOPK)
            with torch.no_grad():
                logits = torch.nn.functional.linear(hidden.float(), module.gate.weight.float())
                probs = torch.softmax(logits, dim=-1)
            return rt.execute(layer, hidden, selected, weights.to(torch.bfloat16), probs)

        return forward

    def execute(self, layer, hidden, selected, weights, probs):
        assert layer == self.index % LAYERS
        diagnostic = self.diag and self.index >= LAYERS
        route_started = time.perf_counter() if diagnostic else None
        if self.index < LAYERS:
            gathered = gather_global_routes(layer, selected, weights, probs,
                                              probability_tail=128)
            self.history.update(layer, gathered.routes.full_router_probs)
            gate = (self.history.sums[layer] /
                    max(1, len(self.history.rows[layer]))).astype(np.float32)
        else:
            gathered = self.metadata.collect(self.index, selected, probs)
            gate = gathered.gate_scores
        routes = gathered.routes
        out = self.policy.apply(self.index, routes.selected_experts,
                                routes.routing_weights, routes.origin_ranks, gate,
                                np.zeros((EXPERTS, self.world), np.int32))
        targets, effective, _, lengths, destinations, fetches, _ = out
        event = plan_rank_partial_layout(effective, lengths, destinations,
                                         routes.origin_ranks, gathered.counts, self.rank,
                                         experts=EXPERTS)
        event['targets'] = targets
        slots = self.native_executor.native.bind_layer_slots(
            self.policy.slots[self.rank, :self.cap], np.arange(self.cap, dtype=np.int32),
            layer, [group[0] for group in event['groups']])
        event['groups'] = [(expert, rows, cols, slot)
                           for (expert, rows, cols), slot in zip(event['groups'], slots)]
        event = pack_rank_partial_layout(event)
        for rank, key, slot, victim, replica in fetches:
            assert not replica
            if rank == self.rank:
                assert self.keys[slot] == victim
                self.h2d.enqueue_demand(slot, key, self.sources[key])
                self.keys[slot] = key
        dense = torch.zeros((hidden.shape[0], EXPERTS), device='cuda',
                            dtype=torch.float32).scatter_add_(
                                1, event['targets'][selected], weights.float()).to(weights.dtype)
        if diagnostic:
            self.diag_route_host_s += time.perf_counter() - route_started
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        pending = self.transport.forward(hidden, dense, event, async_op=True)
        received, received_weights, received_ids = pending.finish()
        if diagnostic:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.diag_spans['dispatch'].append((start, end))
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        parts = self.native_executor.compute(
            self, ('current', received, received_ids, received_weights), event, layer)
        if diagnostic:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.diag_spans['expert'].append((start, end))
            start = torch.cuda.Event(enable_timing=True)
            start.record()
        result = combine_rank_partials(self.transport, hidden, parts, event, torch.bfloat16)
        if diagnostic:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.diag_spans['return'].append((start, end))
        self.index += 1
        return result

    def close(self):
        self.h2d.close()


def generate(model, runtime, ids, count):
    dist.barrier()
    torch.cuda.synchronize()
    release = torch.tensor([time.perf_counter_ns() + 200_000_000 if dist.get_rank() == 0
                            else 0], device='cuda', dtype=torch.int64)
    dist.broadcast(release, 0)
    start = int(release.item())
    while time.perf_counter_ns() < start:
        time.sleep(.0001)
    mask = torch.ones_like(ids)
    past = None
    tokens, stamps = [], []
    diagnostic_steps = [] if runtime.diag else None
    diagnostic_events = [] if runtime.diag else None
    with torch.inference_mode():
        for step in range(count):
            if runtime.diag:
                previous = (runtime.h2d.metrics['bytes'], runtime.native_executor.waits,
                            runtime.native_executor.groups, runtime.native_executor.waves,
                            runtime.diag_route_host_s, runtime.diag_h2d_host_wait_s,
                            runtime.transport.forward_bytes, runtime.transport.return_bytes)
            position = torch.arange(mask.shape[1] - ids.shape[1], mask.shape[1],
                                    device='cuda')[None, :].expand(len(ids), -1)
            out = model(input_ids=ids, attention_mask=mask, position_ids=position,
                        past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            next_ids = out.logits[:, -1].argmax(-1)
            assert torch.isfinite(out.logits).all()
            tokens.append(next_ids)
            torch.cuda.synchronize()
            stamps.append(time.perf_counter_ns())
            if runtime.diag:
                diagnostic_events.append({name: pairs[:] for name, pairs in
                                          runtime.diag_spans.items()})
                for pairs in runtime.diag_spans.values():
                    pairs.clear()
                diagnostic_steps.append(dict(step=step,
                    rank_step_ms=(stamps[-1] - (start if step == 0 else stamps[-2])) / 1e6,
                    route_to_dispatch_host_ms=(runtime.diag_route_host_s - previous[4]) * 1000,
                    h2d_host_wait_ms=(runtime.diag_h2d_host_wait_s - previous[5]) * 1000,
                    h2d_bytes=runtime.h2d.metrics['bytes'] - previous[0],
                    expert_wait_calls=runtime.native_executor.waits - previous[1],
                    expert_groups=runtime.native_executor.groups - previous[2],
                    expert_waves=runtime.native_executor.waves - previous[3],
                    dispatch_bytes=runtime.transport.forward_bytes - previous[6],
                    return_bytes=runtime.transport.return_bytes - previous[7]))
            if step < count - 1:
                ids = next_ids[:, None]
                mask = torch.cat((mask, mask.new_ones((len(ids), 1))), 1)
    assert runtime.index == LAYERS * count
    actual = torch.stack(tokens, 1).cpu().numpy()
    if runtime.diag:
        for row, events in zip(diagnostic_steps, diagnostic_events):
            for name, pairs in events.items():
                row[name + '_ms'] = sum(before.elapsed_time(after)
                                        for before, after in pairs)
    result = dict(release_ns=start, first_ns=stamps[0], end_ns=stamps[-1],
                  output_tokens=count, finite_logits=True,
                  argmax_hash=hashlib.sha256(actual.tobytes()).hexdigest(),
                  tokens=actual.tolist())
    if runtime.diag:
        result['diagnostic_steps'] = diagnostic_steps
    return result


def main(args):
    rank = int(os.environ['RANK'])
    physical = [0, 1, 4, 5]
    assert dist.is_available() and os.environ['MGO_V2_PHYSICAL_GPUS'] == '0,1,4,5'
    cpus = json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except FileNotFoundError:
            pass
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', device_id=torch.device('cuda:0'))
    assert dist.get_world_size() == 4
    manifest = Path(os.environ['MGO_HEADLINE_WORKLOADS'])
    spec = next(x for x in json.loads(manifest.read_text())['cells'] if x['cell'] == args.cell)
    assert spec['model'] == 'DeepSeekV2Lite' and spec['model_path'] == str(MODEL)
    batch = 1 if args.smoke else spec['local_batch']
    percent = spec['cache_percent']
    assert percent in (20, 30, 40, 50)
    slots = 26 * 64 * percent // 100
    expected_per_rank = [slots // 4 + (r < slots % 4) for r in range(4)]
    assert spec['expert_slots'] == slots
    assert spec['expert_slots_per_rank'] == expected_per_rank
    capacities = [x - 2 for x in expected_per_rank]
    assert all(x > 0 for x in capacities)
    model, pool, sources = load_model_and_store()
    runtime = DeepSeekRuntime(batch, capacities, sources, pool)
    runtime.install(model)
    for repeat in range((1 if args.smoke else args.repeats) + 1):
        phase = 'warmup' if repeat == 0 else 'target'
        rows = json.loads(Path(spec[phase]['path']).read_text())['requests']
        local = rows[rank:rank + 1] if args.smoke else rows[rank * batch:(rank + 1) * batch]
        ids = torch.tensor([r['input_ids'][-32:] if args.smoke else r['input_ids']
                            for r in local], device='cuda')
        if repeat:
            runtime.reset()
        assert np.all(runtime.keys < 0)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            write(args.output / 'phase.json', dict(system='main_OURS', phase=phase,
                                                    repeat=repeat, cell=args.cell))
        result = generate(model, runtime, ids, 2 if args.smoke else 64)
        runtime.h2d.synchronize()
        assert np.array_equal(runtime.keys[:runtime.cap],
                              runtime.policy.slots[rank, :runtime.cap])
        result.update(rank=rank, repeat=repeat, phase=phase,
                      request_ids=[r['request_id'] for r in local],
                      expert_executor='native_deepseek', policy='LA_CA_NEAR',
                      prefetch_off=True, cache_start='empty',
                      cache_capacity_slots=runtime.cap,
                      physical_cache_slots=runtime.cache.shape[0],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      pinned_host_bytes=pool.numel() * pool.element_size(),
                      h2d_bytes=runtime.h2d.metrics['bytes'],
                      remote_dispatch_bytes=runtime.transport.forward_bytes,
                      remote_return_bytes=runtime.transport.return_bytes)
        write(args.output / f'repeat{repeat}_rank{rank}.json', result)
        dist.barrier()
        if rank == 0:
            ranked = [json.loads((args.output / f'repeat{repeat}_rank{r}.json').read_text())
                      for r in range(4)]
            assert len({row['release_ns'] for row in ranked}) == 1
            first = max(row['first_ns'] for row in ranked)
            end = max(row['end_ns'] for row in ranked)
            start = ranked[0]['release_ns']
            count = result['output_tokens']
            report = dict(status='PASS', system='main_OURS', repeat=repeat,
                          TTFT=(first - start) / 1e9,
                          TPOT=(end - first) / 1e9 / (count - 1),
                          E2E=(end - start) / 1e9,
                          output_tokens=count, global_requests=4 * batch)
            write(args.output / f'repeat{repeat}.json', report)
            print(json.dumps(report), flush=True)
        dist.barrier()
    runtime.close()
    if rank == 0:
        write(args.output / 'result.json', dict(status='PASS', system='main_OURS',
                                                cell=args.cell, primary_repeats=args.repeats,
                                                headline_eligible=not args.smoke))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=3)
    main(parser.parse_args())
