#!/usr/bin/env python3
"""One physical frozen-action cell; no online placement/replica search."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import benchmark_model as b  # pins rank before importing torch / native extension
os.environ.pop('NCCL_P2P_DISABLE', None)  # bootstrap default must not override frozen conditions
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from frozen_replica_schedule import FrozenReplicaState, digest
import mgo_v2.communicator as comm
from mgo_v2.runtime import gather_global_routes
from mgo_v2.types import LocalExecPlan

PACKET = Path(__file__).resolve().parents[1] / 'experiments/fetch_comm_pareto_p2p_20261002'
INPUTS = Path('/home/hwlee/mgo-results/runtime_validation_20261001')
MODEL = '/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
ACTIVE_RUNTIME = None


class TraceDivergence(RuntimeError):
    pass


class FrozenRuntime:
    def __init__(self, events, capacity, cap, executor, rank):
        self.events = events; self.index = 0; self.rank = rank; self.executor = executor
        self.state = FrozenReplicaState(capacity, cap)
        self.collectives = b.CollectiveStats()
        self.records = []; self.intervals = []; self.current = None
        self.original_exchange = comm._all_to_all_varlen
        comm._all_to_all_varlen = self.exchange

    def exchange(self, send, send_counts, recv_counts, stats=None, kind='payload'):
        if kind in ('dispatch_hidden', 'return_outputs'):
            e = self.events[self.index]
            phase = 'dispatch' if kind == 'dispatch_hidden' else 'combine'
            counts = e[phase + '_counts']
            assert list(send_counts) == counts[self.rank], (self.index, phase, send_counts, counts[self.rank])
            assert list(recv_counts) == [row[self.rank] for row in counts]
            assert send.dtype == b.torch.bfloat16 and send.shape == (sum(send_counts), 2048)
            self.current[phase + '_send_counts'] = list(send_counts)
            self.current[phase + '_recv_counts'] = list(recv_counts)
        return self.original_exchange(send, send_counts, recv_counts, stats, kind)

    def forward_layer(self, layer, hidden_states, selected_experts, routing_weights, full_router_probs, is_decode):
        assert self.index < 432
        e = self.events[self.index]
        assert layer == e['layer'] and is_decode == (self.index >= 48)
        whole_start = time.perf_counter()
        start, end = b.torch.cuda.Event(enable_timing=True), b.torch.cuda.Event(enable_timing=True)
        start.record()
        gathered = gather_global_routes(layer, selected_experts, routing_weights, full_router_probs, self.collectives)
        routes = gathered.routes
        expected = e['_selected']; origins = e['_origins']
        if not b.np.array_equal(routes.selected_experts, expected) or not b.np.array_equal(routes.origin_ranks, origins):
            same_shape = routes.selected_experts.shape == expected.shape
            different = b.np.argwhere(routes.selected_experts != expected).tolist() if same_shape else []
            evidence = dict(status='TRACE_DIVERGENCE', rank=self.rank, event=self.index, step=self.index // 48,
                            layer=layer, expected_shape=list(expected.shape), observed_shape=list(routes.selected_experts.shape),
                            mismatch_count=len(different) if same_shape else None,
                            first_mismatches=[dict(token=t, slot=k, expected=int(expected[t, k]), observed=int(routes.selected_experts[t, k])) for t, k in different[:16]],
                            origins_match=b.np.array_equal(routes.origin_ranks, origins))
            raise TraceDivergence(json.dumps(evidence))
        application_start = time.perf_counter()
        self.state.apply(e)
        offset = gathered.local_offset
        n = hidden_states.shape[0]
        local_routes = [dict(zip(map(int, es), map(float, ws))) for es, ws in
                        zip(routes.selected_experts[offset:offset+n], routes.routing_weights[offset:offset+n])]
        owner = e['_owner'][self.rank]
        plan = e['_plans'][self.rank]
        application_seconds = time.perf_counter() - application_start
        self.current = dict(event=self.index, layer=layer, application_seconds=application_seconds,
                            post_state_sha256=e['post_state_sha256'], raw_route_parity=True)
        received = comm.dispatch_tokens(hidden_states, local_routes, owner, 8, self.collectives)
        # Count actual execution rows and compare the complete token/expert routing
        # before submitting any expert work; set/order equality is exact.
        rows = []
        for origin, es, ds in zip(e['origins'], e['selected'], e['destinations']):
            # Global index is mapped to rank-local token index below.
            rows.append((origin, es, ds))
        expected_rows = set(); token_index = [0] * 4
        for origin, es, ds in rows:
            for expert, dst in zip(es, ds):
                if dst == self.rank: expected_rows.add((origin, token_index[origin], expert))
            token_index[origin] += 1
        rr, ri, re = received.origin_rank.cpu().tolist(), received.origin_index.cpu().tolist(), received.expert_ids.cpu().tolist()
        actual_rows = [(origin, idx, expert) for origin, idx, experts in zip(rr, ri, re) for expert in experts if expert >= 0]
        assert len(actual_rows) == len(expected_rows) and set(actual_rows) == expected_rows
        partials = self.executor.execute(layer, received, plan, is_decode=is_decode)
        actual_cache = sorted(map(list, self.executor.dispatcher.get_cached_slots(0)))
        assert actual_cache == e['rank_cache'][self.rank], (self.index, 'physical cache mismatch')
        cache_stats = self.executor.dispatcher.get_cache_stats().tolist()
        assert cache_stats[3] == self.executor.expected_fetches
        modes = self.executor.dispatcher.get_fetch_mode_counts().tolist()
        assert modes[1] == 0 and modes[0] == cache_stats[3], 'non-direct expert copy detected'
        # Exactly one partial row must return for every scheduled expert route.
        assert partials.values.shape[0] == len(expected_rows)
        returned = [(rr[t], ri[t], e) for t, e in zip(partials.token_indices.cpu().tolist(), partials.expert_ids.cpu().tolist())]
        assert len(returned) == len(expected_rows) and set(returned) == expected_rows
        output = comm.return_partials(partials, received, n, self.collectives)
        end.record(); self.intervals.append((start, end))
        self.current.update(physical_fetches=cache_stats[3], physical_cache_parity=True,
                            exact_execution_rows=len(expected_rows), h2d_classes=e['rank_fetch_classes'][self.rank],
                            layer_host_seconds=time.perf_counter() - whole_start)
        self.records.append(self.current); self.index += 1
        return output


def main(args):
    global ACTIVE_RUNTIME
    assert b.BOOT['visible_gpu'] == str([0, 1, 4, 5][b.BOOT['local_rank']])
    expected_env = {'NCCL_P2P_LEVEL': 'LOC', 'NCCL_IB_DISABLE': '1'} if args.mode == 'R3' else {}
    assert {k: v for k, v in os.environ.items() if k.startswith('NCCL_')} == expected_env
    metadata_path = PACKET / 'physical_fkc_schedule_metadata.json'
    metadata = json.loads(metadata_path.read_text()); meta = metadata['schedules'][args.point]
    schedule = Path(meta['path'])
    assert hashlib.sha256(schedule.read_bytes()).hexdigest() == meta['sha256']
    with gzip.open(schedule, 'rt') as f: events = [json.loads(line) for line in f]
    assert len(events) == 432
    # Parse immutable data and execution containers before primary timing.
    for e in events:
        e['_selected'] = b.np.asarray(e['selected'], dtype=b.np.int64)
        e['_origins'] = b.np.asarray(e['origins'], dtype=b.np.int64)
        e['_owner'] = [{int(k): v for k, v in owner.items()} for owner in e['local_owner']]
        e['_plans'] = [LocalExecPlan(hit_ops=[tuple(x) for x in p['hit_ops']], miss_ops=[tuple(x) for x in p['miss_ops']]) for p in e['local_exec']]
    b.torch.set_num_threads(4); b.torch.cuda.set_device(0)
    b.dist.init_process_group('nccl', device_id=b.torch.device('cuda:0'))
    rank = b.dist.get_rank(); assert b.dist.get_world_size() == 4
    b.warmup_collectives()
    source_meta = metadata['raw_receipts'][rank]
    source_path = Path(source_meta['path']); assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_meta['sha256']
    source = json.loads(source_path.read_text()); expected_tokens = source['generated_tokens']; del source
    model, handle, dispatcher, store = b.load_qwen3_slots(MODEL, str(INPUTS / 'expert_store'))
    assert model.config.hidden_size == 2048 and model.config.moe_intermediate_size * 2048 * 3 * 2 == 9 * 1024**2
    executor = b.LegacySlotExecutorAdapter(dispatcher, metadata['capacities'][rank], 0, 128)
    dispatcher.reset_phase_times()
    runtime = FrozenRuntime(events, metadata['capacities'], meta['duplicate_cap'], executor, rank)
    ACTIVE_RUNTIME = runtime
    assert b.attach_qwen3_runtime(model, runtime) == 48
    tok = b.AutoTokenizer.from_pretrained(MODEL, local_files_only=True, padding_side='left')
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    rows = json.loads((INPUTS / 'screen_workload.json').read_text())[rank*8:(rank+1)*8]
    texts = [tok.apply_chat_template([{'role': 'user', 'content': row['question'] + '\nReturn only the final numeric answer, without explanation.'}], tokenize=False, add_generation_prompt=True) for row in rows]
    encoded = {k: v.cuda() for k, v in tok(texts, padding=True, return_tensors='pt').items()}
    prefill = {}
    def snapshot():
        prefill.update(fetches=executor.expected_fetches, phase_us=dispatcher.get_phase_times().tolist(),
                       fetch_modes=dispatcher.get_fetch_mode_counts().tolist(),
                       payload_bytes=dict(runtime.collectives.payload_bytes), collective_events=len(runtime.collectives.events))
    b.dist.barrier(); b.torch.cuda.synchronize()
    print(json.dumps(dict(started=True, mode=args.mode, point=args.point, rank=rank)), flush=True)
    generation_start = time.perf_counter()
    with b.torch.inference_mode():
        output, seconds = b.generate(model, encoded['input_ids'], encoded['attention_mask'], 9, snapshot)
    b.torch.cuda.synchronize(); generation_seconds = time.perf_counter() - generation_start
    assert runtime.index == 432
    tokens = output.cpu().tolist()
    assert tokens == expected_tokens, 'GENERATED_TOKEN_MISMATCH against exact capture'
    phase_us = dispatcher.get_phase_times().tolist()
    summary = runtime.collectives.summary()
    interval = dict(prefill={}, decode={})
    for i, (kind, start, end) in enumerate(runtime.collectives.events):
        phase = 'prefill' if i < prefill['collective_events'] else 'decode'
        interval[phase][kind] = interval[phase].get(kind, 0) + start.elapsed_time(end)
    moe_intervals = [s.elapsed_time(e) for s, e in runtime.intervals]
    for record, ms in zip(runtime.records, moe_intervals): record['moe_cuda_interval_ms'] = ms
    receipt = dict(status='PASS', mode=args.mode, point=args.point, rho=meta['rho'], rank=rank, boot=b.BOOT,
                   schedule_sha256=meta['sha256'], transport_env=expected_env, nccl_info_in_timing=False,
                   model_warmup_forwards=0, prefill_forwards=1, decode_forwards=8,
                   rank_step_seconds=seconds, rank_generation_seconds=generation_seconds,
                   generated_tokens=tokens, generated_tokens_equal_source=True, events=runtime.records,
                   prefill=prefill, physical_fetches=executor.expected_fetches,
                   cache_stats=dispatcher.get_cache_stats().tolist(), phase_us=phase_us,
                   phase_us_labels=['fetch_wait_host_us', 'weight_bind_host_us', 'compute_and_sync_host_us', 'combine_host_us'],
                   fetch_modes=dispatcher.get_fetch_mode_counts().tolist(), collectives=summary,
                   collective_intervals_ms=interval, checkpoint=store['identity'],
                   timing_scope='Includes frozen action application, per-event trace/cache validation and lightweight intervals; excludes load/tokenization/schedule generation/warmup',
                   omitted_timings=['isolated H2D GPU duration', 'isolated expert GPU kernel duration'],
                   input_workload_sha256=hashlib.sha256((INPUTS / 'screen_workload.json').read_bytes()).hexdigest())
    (args.output / f'rank{rank}.json').write_text(json.dumps(receipt, separators=(',', ':')) + '\n')
    print(json.dumps(dict(status='PASS', rank=rank, point=args.point, mode=args.mode, decode_seconds=sum(seconds[1:]))), flush=True)
    b.dist.barrier(); b.dist.destroy_process_group(); b.detach_qwen3_runtime(model)
    del model, runtime, executor, dispatcher; handle.clean_up_resources()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', choices=['T0', 'R3'], required=True)
    parser.add_argument('--point', choices=['F', 'K', 'C'], required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args)
    except BaseException as exc:
        failure = dict(status='TRACE_DIVERGENCE' if isinstance(exc, TraceDivergence) else 'FAIL_CORRECTNESS_OR_EXECUTION',
                       rank=b.BOOT['local_rank'], point=args.point, mode=args.mode, error=str(exc), traceback=traceback.format_exc())
        if ACTIVE_RUNTIME is not None:
            failure['completed_events'] = ACTIVE_RUNTIME.index
            failure['completed_event_receipts'] = ACTIVE_RUNTIME.records
        (args.output / f'failure-rank{b.BOOT["local_rank"]}.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise
