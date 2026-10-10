"""Summarize the bounded, frozen-route one-decode-step model pilot."""

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/single_step_locality_pilot_20261010/jobs')
JOBS = {
    'BR': RAW / 'r4_n_br_c30_16_diagnostic_v5',
    'Near': RAW / 'r4_n_br_c30_16_diagnostic_v6',
    'CA_NATIVE': RAW / 'r4_n_br_c30_16_diagnostic_v7',
    'LA_CA': RAW / 'r4_n_br_c30_16_diagnostic_v8',
}


def read(path):
    return json.loads(path.read_text())


def phase_ms(diagnostic, name):
    return sum(x['stream_seconds'] * 1000 for x in diagnostic['segments']
               if x['event_index'] >= 48 and x['phase'] == name)


def summarize(label, job):
    status = read(job / 'status.json')
    assert status['status'] == 'PASS' and not status['remaining_gpu_processes']
    assert sorted(x['gpu'] for x in status['restored_model_loads']) == [0, 1, 4, 5]
    samples = [read(job / f'repeat{i}.json')['TPOT'] for i in range(1, 5)]
    ranks = [read(job / f'generation_diagnostic_rank{rank}.json') for rank in range(4)]
    assert all(x['status'] == 'PASS' and x['token_and_cache_parity'] for x in ranks)
    route_hashes = [x['decode_route_sha256'] for x in ranks]
    assert len(set(route_hashes)) == 1
    placements = ranks[0]['decode_placement_events']
    assert len(placements) == 48
    local = sum(x['local_expert_rows'] for x in placements)
    all_rows = sum(x['total_expert_rows'] for x in placements)
    assert local + sum(x['remote_expert_rows'] for x in placements) == all_rows
    copies = [sum(x['demand_misses'] for x in rank['decode_cache_events']) for rank in ranks]
    rows = [sum(x['token_expert_uses'] for x in rank['decode_cache_events']) for rank in ranks]
    planned = [sum(x['rank_fetches'][rank] for x in placements) for rank in range(4)]
    assert copies == planned and sum(rows) == all_rows
    forward = sum(x['decode_transport_bytes']['forward_bytes'] for x in ranks)
    returned = sum(x['decode_transport_bytes']['return_bytes'] for x in ranks)
    targets = [[read(job / f'repeat1_rank{rank}.json')['tokens'][t][0]
                for t in range(16)] for rank in range(4)]
    for i in range(1, 5):
        for rank in range(4):
            result = read(job / f'repeat{i}_rank{rank}.json')
            assert result['no_compile'] and result['prefetch_off']
            assert result['grouped_decode_mode'] == 'hit_then_miss'
            assert result['decode_policy'] == {'BR':'BR', 'Near':'LA_CA_NEAR', 'CA_NATIVE':'CA_NATIVE', 'LA_CA':'LA_CA'}[label]
            assert [t[0] for t in result['tokens']] == targets[rank]
    return dict(policy=label,job=str(job),source_commit=status['source_commit'],
                workload_sha256=status['workload_sha256'],route_sha256=route_hashes[0],
                first_token_sha256=hashlib.sha256(json.dumps(targets).encode()).hexdigest(),
                TPOT_samples_seconds=samples,TPOT_median_seconds=statistics.median(samples),
                TPOT_range_seconds=[min(samples),max(samples)],
                local_expert_rows=local,total_expert_rows=all_rows,local_expert_row_fraction=local/all_rows,
                demand_copies_by_rank=copies,max_rank_copies=max(copies),
                expert_rows_by_rank=rows,max_rank_expert_rows=max(rows),
                forward_peer_bytes=forward,return_peer_bytes=returned,total_peer_bytes=forward+returned,
                diagnostic_max_rank_exposed_H2D_ms=max(phase_ms(x,'required_h2d_exposed_wait') for x in ranks),
                diagnostic_max_rank_expert_ms=max(phase_ms(x,'moe.expert_compute')+phase_ms(x,'expert_grouped_gemm_kernels') for x in ranks),
                diagnostic_max_rank_dispatch_ms=max(sum(phase_ms(x,p) for p in ('moe.forward_a2a','forward_token_a2a_submit','moe.forward_complete')) for x in ranks),
                diagnostic_max_rank_return_ms=max(phase_ms(x,'moe.return_a2a')+phase_ms(x,'return_token_a2a') for x in ranks))


def main():
    data = {name:summarize(name,path) for name,path in JOBS.items()}
    assert len({x['route_sha256'] for x in data.values()}) == 1
    assert len({x['first_token_sha256'] for x in data.values()}) == 1
    assert len({x['workload_sha256'] for x in data.values()}) == 1
    reference=data['BR']
    for row in data.values():
        row['peer_change_vs_BR_pct']=100*(row['total_peer_bytes']/reference['total_peer_bytes']-1)
        row['TPOT_change_vs_BR_pct']=100*(row['TPOT_median_seconds']/reference['TPOT_median_seconds']-1)
        row['max_copies_change_vs_BR']=row['max_rank_copies']-reference['max_rank_copies']
        row['max_expert_rows_change_vs_BR']=row['max_rank_expert_rows']-reference['max_rank_expert_rows']
    (HERE/'SUMMARY.json').write_text(json.dumps(dict(status='PASS',rows=data),indent=2)+'\n')
    print(json.dumps({name:{k:row[k] for k in ('TPOT_samples_seconds','local_expert_row_fraction','max_rank_copies','max_rank_expert_rows','total_peer_bytes','TPOT_change_vs_BR_pct','peer_change_vs_BR_pct')} for name,row in data.items()},indent=2))


if __name__=='__main__':main()
