"""Summarize exclusive full-decode diagnostic spans for the R4 A/B/C study."""

import json
from collections import defaultdict
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
OUT = PKG / 'experiments/expert_grouping_ablation_20261009'
ROOT = Path('/home/hwlee/mgo-results/expert_grouping_ablation_20261009/jobs')
PHYSICAL = (0, 1, 4, 5)
GROUP = {
    'attention_dense_residual': 'Attention, dense and output',
    'router_gate_compute': 'Router gate',
    'moe.metadata': 'Routing metadata',
    'metadata_gpu_packet_pack': 'Routing metadata',
    'metadata_rank_all_gather': 'Routing metadata',
    'metadata_device_to_host': 'Routing metadata',
    'metadata_cpu_parse': 'Routing metadata',
    'metadata_gate_history': 'Routing metadata',
    'metadata_demand_histogram': 'Routing metadata',
    'moe.current_controller': 'Placement and index',
    'placement_controller_cpu': 'Placement and index',
    'layout_cpu': 'Placement and index',
    'layout_device_materialization': 'Placement and index',
    'moe.demand_h2d': 'Demand H2D submission',
    'required_h2d_exposed_wait': 'Exposed H2D wait',
    'moe.forward_a2a': 'Forward dispatch',
    'forward_token_a2a_submit': 'Forward dispatch',
    'moe.forward_complete': 'Forward dispatch',
    'moe.expert_compute': 'Expert execution and preparation',
    'expert_grouped_gemm_kernels': 'Expert execution and preparation',
    'moe.return_a2a': 'Return and combine',
    'return_token_a2a': 'Return and combine',
    'moe_other': 'MoE runtime residual',
}
ORDER = (
    'Attention, dense and output', 'Router gate', 'Routing metadata',
    'Placement and index', 'Demand H2D submission', 'Forward dispatch',
    'Exposed H2D wait', 'Expert execution and preparation',
    'Return and combine', 'MoE runtime residual',
)


def read(path):
    return json.loads(path.read_text())


def audit_rank(job, arm, rank, expected_tokens):
    data = read(job / f'generation_diagnostic_rank{rank}.json')
    assert data['status'] == 'PASS' and data['token_and_cache_parity']
    assert data['output']['tokens'] == expected_tokens
    assert len(data['decode_cache_events']) == 48*63
    assert data['validation']['status'] == 'PASS'
    segments = data['segments']
    markers = [(index, row['token_ready_index']) for index,row in enumerate(segments)
               if 'token_ready_index' in row]
    assert [token for _,token in markers] == list(range(64))
    window = segments[markers[0][0]:markers[-1][0]]
    assert window
    phase = defaultdict(float)
    for row in window:
        phase[row['phase']] += row['stream_seconds']
    unknown = set(phase) - set(GROUP)
    assert not unknown, (arm, rank, unknown)
    seconds = sum(phase.values())
    wall = (data['output']['end_ns'] - data['output']['first_ns']) / 1e9
    assert abs(seconds - wall) < .05, (seconds, wall)
    category = defaultdict(float)
    for name,value in phase.items():
        category[GROUP[name]] += value
    assert abs(sum(category.values())-seconds) < 1e-5
    return dict(rank=rank, physical_gpu=PHYSICAL[rank], decode_seconds=wall,
                event_seconds=seconds, phase_seconds=dict(phase),
                category_seconds=dict(category),
                diagnostic_h2d_service_seconds=data['h2d_service_seconds'],
                diagnostic_h2d_bytes=sum(row['bytes'] for row in data['h2d_copies']),
                decoded_expert_uses=sum(row['expert_uses'] for row in data['decode_cache_events']),
                decoded_main_hits=sum(row['main_hits'] for row in data['decode_cache_events']),
                decoded_demand_misses=sum(row['demand_misses'] for row in data['decode_cache_events']))


def main():
    all_arms = {}
    for arm in 'abc':
        job = ROOT / f'r4_{arm}_diagnostic_v1'
        state = read(job/'status.json')
        assert state['status'] == 'PASS' and state['kind'] == 'diagnostic'
        assert len(state['restored_model_loads']) == 8
        primary = ROOT / f'r4_{arm}_full_v1'
        ranks = []
        for rank in range(4):
            expected = read(primary/f'repeat1_rank{rank}.json')['tokens']
            measured = read(job/f'repeat1_rank{rank}.json')
            assert measured['tokens'] == expected
            assert measured['validation']['scheduler']['bytes'] == read(primary/f'repeat1_rank{rank}.json')['validation']['scheduler']['bytes']
            assert measured['validation']['state_hash'] == read(primary/f'repeat1_rank{rank}.json')['validation']['state_hash']
            ranks.append(audit_rank(job, arm, rank, expected))
        critical = max(ranks, key=lambda row: row['decode_seconds'])
        seconds = critical['event_seconds']
        category = {name:dict(ms_per_token=critical['category_seconds'].get(name,0)*1000/63,
                              percent=critical['category_seconds'].get(name,0)/seconds*100)
                    for name in ORDER}
        fine = {name:dict(ms_per_token=value*1000/63,percent=value/seconds*100)
                for name,value in sorted(critical['phase_seconds'].items())}
        assert abs(sum(row['percent'] for row in category.values())-100) < 1e-5
        all_arms[arm] = dict(job=str(job), critical_rank=critical['rank'],
                             critical_physical_gpu=critical['physical_gpu'],
                             diagnostic_decode_seconds=seconds,
                             diagnostic_tpot_ms=seconds*1000/63,
                             category=category, fine_phase=fine, ranks=ranks)
    out = dict(status='PASS', basis='one separately instrumented decode64 target per arm; first-token-ready to last-token-ready; exclusive current-stream partition on slowest diagnostic rank',
               arms=all_arms, note='H2D copy-stream service overlaps current-stream phases and is not added to 100%. NCCL spans include peer-arrival wait; they are not pure wire times. Instrumentation inflates total TPOT and must not replace primary timings.')
    (OUT/'BREAKDOWN.json').write_text(json.dumps(out,indent=2)+'\n')
    lines = ['# R4 decode critical-path breakdown', '',
             'This is a separate, instrumented full 64-token target on Qwen3-30B ShareGPT R4/C30/B16/input512. Each arm reproduces its unprofiled target tokens, H2D bytes and final cache state. The window starts when the first token is ready and ends when the last token is ready. Values below are exclusive GPU current-stream spans on the slowest diagnostic rank; each column sums to 100%.',
             '',
             '| Component | A ms/token | A % | B ms/token | B % | C ms/token | C % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name in ORDER:
        values=[]
        for arm in 'abc':
            entry=all_arms[arm]['category'][name]
            values.extend((f'{entry["ms_per_token"]:.2f}',f'{entry["percent"]:.1f}%'))
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines.append('| **Instrumented total** | '+ ' | '.join(f'{all_arms[arm]["diagnostic_tpot_ms"]:.2f} | 100%' for arm in 'abc')+' |')
    lines += ['',
              'The grouped GEMM kernel is included in Expert execution and preparation. Required H2D waiting is separate. Demand H2D submission is host/current-stream setup, while H2D copy-stream service runs concurrently and must not be added to the above total.',
              '',
              '| Fine phase | A ms/token | A % | B ms/token | B % | C ms/token | C % |',
              '|---|---:|---:|---:|---:|---:|---:|']
    names=sorted(set().union(*(entry['fine_phase'] for entry in all_arms.values())))
    for name in names:
        values=[]
        for arm in 'abc':
            entry=all_arms[arm]['fine_phase'].get(name,dict(ms_per_token=0,percent=0))
            values.extend((f'{entry["ms_per_token"]:.2f}',f'{entry["percent"]:.1f}%'))
        lines.append('| `'+name+'` | '+' | '.join(values)+' |')
    lines += ['',
              'The metadata all-gather and token return include waiting for other ranks. Their event spans do not establish pure network transmission time. The separately instrumented total is higher than the unprofiled TPOT because the diagnostic adds event markers and tracing. Use PRIMARY_RESULTS.md for the performance comparison.',
              '']
    (OUT/'BREAKDOWN.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'PASS','diagnostic_tpot_ms':{arm:round(entry['diagnostic_tpot_ms'],2) for arm,entry in all_arms.items()}}))


if __name__ == '__main__':
    main()
