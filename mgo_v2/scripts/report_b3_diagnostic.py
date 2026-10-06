"""B3 mechanism-only paired summary. Never substitute capture time for TPOT."""
import csv, json
from pathlib import Path
import numpy as np
from report_b2 import export
from prepare_critical_microbench import PACKET, ROOT, write, sha


def report(cache='C30'):
    root=ROOT/'b3'
    state=json.loads((root/'status.json').read_text())
    assert state['status']=='DIAGNOSTICS_COMPLETE'
    summaries=[];signatures=[];correctness=[];sources={}
    for path in state['completed']:
        cap=Path(path);case=json.loads((cap/'case.json').read_text())
        if case['b3_cache']!=cache:continue
        ranks=export(cap)
        row=dict(cache=cache,policy=case['policy'],executor=case['b3_executor'],capture=str(cap))
        rows=[e for r in ranks for e in r['events']]
        for metric,phase,kind in [
            ('host_expert_loop_ms','expert_loop','host_union_ms'),
            ('host_kernel_launch_ms','expert_compiled_kernel','host_union_ms'),
            ('gpu_expert_service_including_scratch_ms','expert_compiled_kernel','gpu_union_ms'),
            ('gpu_gather_ms','expert_gather','gpu_union_ms'),
            ('gpu_weight_ms','expert_weight_partial','gpu_union_ms'),
            ('forward_nccl_residency_ms','forward_collective_host_call','gpu_union_ms'),
            ('return_nccl_residency_ms','return_collective_host_call','gpu_union_ms')]:
            row[metric]=sum(e['phases'][phase][kind] for e in rows)/32
        row['h2d_wait_upper_ms']=sum(b['gpu_wait_upper_bound_ms'] for e in rows for b in e['slot_wait_bounds'])/32
        for direction in ('forward','return'):
            for name,field in [('host_entry','host_start_ns'),('gpu_start','gpu_start_ns')]:
                row[f'{direction}_{name}_spread_ms']=sum(
                    np.ptp([r['events'][i]['phases'][direction+'_collective_host_call'][field] for r in ranks])/1e6 for i in range(384))/8
        row['clock_p99_us_max']=max(r['clock_alignment']['p99_anchor_residual_us'] for r in ranks)
        assert row['gpu_expert_service_including_scratch_ms']>0,'graph kernel attribution missing'
        for rank in range(4):
            receipt=json.loads((cap/f'rank{rank}.json').read_text());assert receipt['status']=='PASS'
            for filename in (f'rank{rank}.json',f'b2_rank{rank}_analysis.json',f'communication_rank{rank}.json',f'copy_trace_rank{rank}.json',f'profile_rank{rank}.nsys-rep'):
                sources[str(cap/filename)]=sha(cap/filename)
            if case['b3_executor']=='H1':
                corr=json.loads((cap/f'b3_correctness_rank{rank}.json').read_text());assert corr['status']=='PASS'
                correctness.append(dict(cache=cache,policy=case['policy'],rank=rank,**corr))
                signatures.append(dict(cache=cache,policy=case['policy'],rank=rank,**json.loads((cap/f'b3_signatures_rank{rank}.json').read_text())))
        summaries.append(row)
    assert len(summaries)==4
    gates={}
    for policy in ('BR','FCA'):
        a=next(r for r in summaries if r['policy']==policy and r['executor']=='H0')
        b=next(r for r in summaries if r['policy']==policy and r['executor']=='H1')
        host=1-b['host_kernel_launch_ms']/a['host_kernel_launch_ms']
        loop=1-b['host_expert_loop_ms']/a['host_expert_loop_ms']
        parity=True
        for rank in range(4):
            ar=json.loads((Path(a['capture'])/f'rank{rank}.json').read_text())
            br=json.loads((Path(b['capture'])/f'rank{rank}.json').read_text())
            for field in ('controller_counters','decode_expert_copies','decode_expert_bytes','transport_calls'):
                parity &= ar[field]==br[field]
            ac=json.loads((Path(a['capture'])/f'communication_rank{rank}.json').read_text())
            bc=json.loads((Path(b['capture'])/f'communication_rank{rank}.json').read_text())
            parity &= ac['events']==bc['events']
        gates[policy]=dict(host_call_reduction=host,host_loop_reduction=loop,
                          counters_and_packets_equal=bool(parity),
                          diagnostic_gate=bool(host>=.70 and loop>=.40 and parity))
    write(PACKET/'B3_GRAPH_SIGNATURES.json',dict(rows=signatures))
    write(PACKET/'B3_CORRECTNESS.json',dict(status='PASS' if all(g['counters_and_packets_equal'] for g in gates.values()) else 'FAIL',rows=correctness))
    result=dict(status='DIAGNOSTIC_PASS' if all(g['diagnostic_gate'] for g in gates.values()) else 'DIAGNOSTIC_GATE_FAIL',rows=summaries,gates=gates,
                units='ms/step; host and GPU spans overlap. Expert/NCCL residency mean across ranks; arrival spreads max-min across ranks.',
                primary_timing=False,graph_service_note='H1 launch range includes scratch copy, graph replay and graph output copy.',sources=sources)
    write(PACKET/'B3_DIAGNOSTIC_RESULTS.json',result)
    with (PACKET/'B3_DIAGNOSTIC_RESULTS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    return result
if __name__=='__main__':print(json.dumps(report()['gates'],indent=2))
