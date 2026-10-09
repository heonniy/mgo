"""Audit a guarded R4 run and partition its full-decode critical path."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


PHYSICAL = (0, 1, 4, 5)
GROUPS = {
    'attention_dense_residual': 'Attention/dense/output',
    'router_gate_compute': 'Router gate',
    'moe.metadata': 'Routing metadata',
    'metadata_counts': 'Routing metadata',
    'device_to_host_counts': 'Routing metadata',
    'metadata_selected_ids': 'Routing metadata',
    'metadata_routing_weights': 'Routing metadata',
    'metadata_probability_history': 'Routing metadata',
    'device_to_host_controller_materialization': 'Routing metadata',
    'metadata_gpu_packet_pack': 'Routing metadata',
    'metadata_rank_all_gather': 'Routing metadata',
    'metadata_device_to_host': 'Routing metadata',
    'metadata_cpu_parse': 'Routing metadata',
    'metadata_gate_history': 'Routing metadata',
    'metadata_demand_histogram': 'Routing metadata',
    'moe.current_controller': 'Placement/index',
    'placement_controller_cpu': 'Placement/index',
    'layout_cpu': 'Placement/index',
    'layout_device_materialization': 'Placement/index',
    'moe.dense_weights': 'Routing-weight tensor',
    'moe.demand_h2d': 'H2D submission',
    'required_h2d_exposed_wait': 'Exposed H2D wait',
    'moe.h2d_global_barrier': 'Other synchronization',
    'global_barrier_wait': 'Other synchronization',
    'moe.forward_a2a': 'Dispatch/forward',
    'forward_token_a2a_submit': 'Dispatch/forward',
    'moe.forward_complete': 'Dispatch/forward',
    'moe.expert_compute': 'Expert execution',
    'expert_grouped_gemm_kernels': 'Expert execution',
    'moe.return_a2a': 'Return/combine',
    'return_token_a2a': 'Return/combine',
    'moe_other': 'Other MoE runtime',
}


def read(path):
    return json.loads(path.read_text())


def rank_summary(primary, diagnostic, rank):
    base = read(primary/f'repeat1_rank{rank}.json')
    timed = read(primary/f'repeat2_rank{rank}.json')
    profile = read(diagnostic/f'generation_diagnostic_rank{rank}.json')
    diagnosis = read(diagnostic/f'repeat1_rank{rank}.json')
    assert profile['status'] == 'PASS' and profile['token_and_cache_parity']
    assert base['tokens'] == timed['tokens'] == diagnosis['tokens'] == profile['output']['tokens']
    assert base['validation']['state_hash'] == timed['validation']['state_hash'] == diagnosis['validation']['state_hash']
    assert base['validation']['scheduler']['bytes'] == timed['validation']['scheduler']['bytes'] == diagnosis['validation']['scheduler']['bytes']
    events = profile['decode_cache_events']
    assert len(events) == 48*63
    markers = [i for i, row in enumerate(profile['segments']) if 'token_ready_index' in row]
    assert [profile['segments'][i]['token_ready_index'] for i in markers] == list(range(64))
    window = profile['segments'][markers[0]:markers[-1]]
    fine = defaultdict(float)
    for row in window:
        fine[row['phase']] += row['stream_seconds']
    unknown = set(fine) - set(GROUPS)
    assert not unknown, unknown
    phase = defaultdict(float)
    for name, value in fine.items():
        phase[GROUPS[name]] += value
    decode_seconds = sum(phase.values())
    elapsed = (profile['output']['end_ns']-profile['output']['first_ns'])/1e9
    assert abs(decode_seconds-elapsed) < .05
    counters = {key: sum(event[key] for event in events) for key in (
        'expert_uses', 'token_expert_uses', 'main_hits', 'prefetch_hits',
        'demand_misses', 'main_hit_tokens', 'miss_tokens')}
    assert counters['expert_uses'] == counters['main_hits']+counters['prefetch_hits']+counters['demand_misses']
    copies = profile['h2d_copies']
    if copies and all(row.get('event_index') is not None for row in copies):
        decode_copies = [row for row in copies if row['event_index'] >= 48]
        copy_method = 'exact tagged decode DMA service'
    else:
        prefill = read(diagnostic/f'post_diagnostic_rank{rank}.json')
        decode_count = counters['demand_misses']
        decode_copies = copies[-decode_count:] if decode_count else []
        assert len(decode_copies) == decode_count
        assert len(copies)-len(decode_copies) == len(prefill['h2d_copies'])
        copy_method = 'decode tail of trace; prefill count checked against independent one-token pass'
    assert len(decode_copies) == counters['demand_misses']
    dma = sum(row['service_seconds'] for row in decode_copies)
    bytes_ = sum(row['bytes'] for row in decode_copies)
    prefill = read(diagnostic/f'post_diagnostic_rank{rank}.json') if (diagnostic/f'post_diagnostic_rank{rank}.json').exists() else None
    prefill_phase = defaultdict(float)
    if prefill:
        for name, value in prefill['exclusive_stream_partition'].items():
            prefill_phase[GROUPS.get(name, name)] += value
    return dict(rank=rank, physical_gpu=PHYSICAL[rank], primary_tpot_samples=[
        read(primary/f'repeat{i}.json')['TPOT'] for i in (1,2)],
        diagnostic_tpot_seconds=decode_seconds/63,
        phase_seconds=dict(phase), fine_seconds=dict(fine),
        event_counts=counters,
        decode_copy_count=len(decode_copies), decode_copy_bytes=bytes_,
        decode_copy_service_seconds=dma,
        mean_copy_service_ms=1000*dma/len(decode_copies) if decode_copies else 0,
        decode_transport_bytes=profile.get('decode_transport_bytes'),
        copy_method=copy_method,
        prefill_phase_seconds=dict(prefill_phase) if prefill else None,
        prefill_fine_seconds=prefill['exclusive_stream_partition'] if prefill else None,
        grouped_executor_counts=base['grouped_executor_counts'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for root, kind in ((args.primary, 'full'), (args.diagnostic, 'diagnostic')):
        receipt = read(root/'status.json')
        assert receipt['status'] == 'PASS' and receipt['kind'] == kind
        assert len(receipt['restored_model_loads']) == 8
    ranks = [rank_summary(args.primary, args.diagnostic, rank) for rank in range(4)]
    critical = max(ranks, key=lambda row: row['phase_seconds'].get('Expert execution', 0))
    total = sum(critical['phase_seconds'].values())
    assert total > 0
    result = dict(status='PASS', primary=str(args.primary), diagnostic=str(args.diagnostic),
                  anchor_selection='rank with the longest expert-execution span; collective boundaries align total elapsed across ranks',
                  anchor_rank=critical['rank'], anchor_physical_gpu=critical['physical_gpu'],
                  anchor_phase_ms_per_token={k:v*1000/63 for k,v in critical['phase_seconds'].items()},
                  anchor_phase_percent={k:v/total*100 for k,v in critical['phase_seconds'].items()},
                  anchor_fine_ms_per_token={k:v*1000/63 for k,v in critical['fine_seconds'].items()},
                  ranks=ranks,
                  caveat='Diagnostic current-stream spans include peer/host waits and are inflated by instrumentation. DMA service overlaps and is not part of the 100% partition.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status='PASS', anchor_gpu=result['anchor_physical_gpu'],
                          diagnostic_tpot_ms=critical['diagnostic_tpot_seconds']*1000,
                          decode_copies=[row['decode_copy_count'] for row in ranks])))


if __name__ == '__main__':
    main()
