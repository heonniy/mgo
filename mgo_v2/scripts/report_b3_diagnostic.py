"""B3 mechanism-only paired summary. Never substitute capture time for TPOT."""
import csv, json, hashlib, sqlite3, bisect
from collections import defaultdict
from pathlib import Path
import numpy as np
from report_b2 import export
from prepare_critical_microbench import PACKET, ROOT, write, sha


def scratch_dma_ms(capture,rank):
    db=sqlite3.connect((capture/f'interval_analysis/rank{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'));ranges=defaultdict(list)
    for r in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end>start'):
        text=strings.get(r['textId'],r['text']) or ''
        if text.startswith('b2|') and text.endswith('|expert_compiled_kernel'):
            ranges[r['globalTid']].append((r['start'],r['end']))
    ranges={tid:sorted(rs) for tid,rs in ranges.items()};starts={tid:[a for a,b in rs] for tid,rs in ranges.items()}
    runtime={};tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table in tables:
            for r in db.execute(f'select start,end,globalTid,correlationId from {table}'):
                runtime.setdefault((r['globalTid'] & 0xFFFFFFFFFF000000,r['correlationId']),(r['start'],r['end'],r['globalTid']))
    kinds={r['id']:r['label'].lower().replace(' ','').replace('-','') for r in db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')}
    total=0;count=0
    for r in db.execute('select start,end,copyKind,globalPid,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY'):
        if kinds[r['copyKind']] not in ('dtod','d2d','devicetodevice'):continue
        api=runtime.get((r['globalPid'],r['correlationId']))
        if not api:continue
        a,b,tid=api
        if tid not in ranges:continue
        i=bisect.bisect_right(starts[tid],a)-1
        if i>=0 and ranges[tid][i][1]>=b:total+=r['end']-r['start'];count+=1
    db.close()
    return total/1e6,count


def report(cache='C30',candidate='H1'):
    root=ROOT/'b3'
    state=json.loads((root/('h1b_status.json' if candidate=='H1b' else 'status.json')).read_text())
    prefix='B3_H1b' if candidate=='H1b' else 'B3'
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
            ('host_gather_ms','expert_gather','host_union_ms'),
            ('host_weight_ms','expert_weight_partial','host_union_ms'),
            ('host_ready_select_including_wait_ms','expert_ready_select','host_union_ms'),
            ('host_ready_wait_ms','expert_ready_wait','host_union_ms'),
            ('host_record_use_ms','expert_record_use','host_union_ms'),
            ('gpu_expert_kernel_ms','expert_compiled_kernel','gpu_union_ms'),
            ('gpu_gather_ms','expert_gather','gpu_union_ms'),
            ('gpu_weight_ms','expert_weight_partial','gpu_union_ms'),
            ('forward_nccl_residency_ms','forward_collective_host_call','gpu_union_ms'),
            ('return_nccl_residency_ms','return_collective_host_call','gpu_union_ms')]:
            row[metric]=sum(e['phases'].get(phase,{}).get(kind,0.) for e in rows)/32
        dma=[scratch_dma_ms(cap,r) for r in range(4)]
        row['graph_scratch_dma_ms']=sum(x[0] for x in dma)/32
        row['graph_scratch_dma_count']=sum(x[1] for x in dma)
        row['h2d_wait_upper_ms']=sum(b['gpu_wait_upper_bound_ms'] for e in rows for b in e['slot_wait_bounds'])/32
        for direction in ('forward','return'):
            for name,field in [('host_entry','host_start_ns'),('gpu_start','gpu_start_ns')]:
                row[f'{direction}_{name}_spread_ms']=sum(
                    np.ptp([r['events'][i]['phases'][direction+'_collective_host_call'][field] for r in ranks])/1e6 for i in range(384))/8
        row['clock_p99_us_max']=max(r['clock_alignment']['p99_anchor_residual_us'] for r in ranks)
        assert row['gpu_expert_kernel_ms']>0,'graph kernel attribution missing'
        for rank in range(4):
            receipt=json.loads((cap/f'rank{rank}.json').read_text());assert receipt['status']=='PASS'
            for filename in (f'rank{rank}.json',f'b2_rank{rank}_analysis.json',f'communication_rank{rank}.json',f'copy_trace_rank{rank}.json',f'profile_rank{rank}.nsys-rep'):
                sources[str(cap/filename)]=sha(cap/filename)
            if case['b3_executor']==candidate:
                corr=json.loads((cap/f'b3_correctness_rank{rank}.json').read_text());assert corr['status']=='PASS'
                signature_list=corr['graph'].pop('signatures');corr['graph'].update(signature_count=len(signature_list),signatures_sha256=hashlib.sha256(json.dumps(signature_list,separators=(',',':')).encode()).hexdigest())
                correctness.append(dict(cache=cache,policy=case['policy'],rank=rank,**corr))
                signatures.append(dict(cache=cache,policy=case['policy'],rank=rank,**json.loads((cap/f'b3_signatures_rank{rank}.json').read_text())))
        summaries.append(row)
    assert len(summaries)==4
    gates={}
    for policy in ('BR','FCA'):
        a=next(r for r in summaries if r['policy']==policy and r['executor']=='H0')
        b=next(r for r in summaries if r['policy']==policy and r['executor']==candidate)
        host=1-b['host_kernel_launch_ms']/a['host_kernel_launch_ms']
        loop=1-b['host_expert_loop_ms']/a['host_expert_loop_ms']
        parity=True
        kernel_identity=[]
        for rank in range(4):
            ax=json.loads((Path(a['capture'])/f'b2_rank{rank}_analysis.json').read_text())
            bx=json.loads((Path(b['capture'])/f'b2_rank{rank}_analysis.json').read_text())
            ah=ax['kernel_names']['expert_compiled_kernel'];bh=bx['kernel_names']['expert_compiled_kernel']
            same=all(bh.get(name)==count for name,count in ah.items())
            kernel_identity.append(dict(rank=rank,all_h0_kernel_counts_preserved=same,h0=ah,h1=bh))
            assert same, ('Graph kernel attribution or compiled-kernel identity needs diagnosis',policy,rank,ah,bh)

            ar=json.loads((Path(a['capture'])/f'rank{rank}.json').read_text())
            br=json.loads((Path(b['capture'])/f'rank{rank}.json').read_text())
            for field in ('controller_counters','decode_expert_copies','decode_expert_bytes','transport_calls'):
                parity &= ar[field]==br[field]
            ac=json.loads((Path(a['capture'])/f'communication_rank{rank}.json').read_text())
            bc=json.loads((Path(b['capture'])/f'communication_rank{rank}.json').read_text())
            parity &= ac['events']==bc['events']
        gates[policy]=dict(host_call_reduction=host,host_loop_reduction=loop,
                          counters_and_packets_equal=bool(parity),kernel_identity=kernel_identity,
                          diagnostic_gate=bool(host>=.70 and loop>=.40 and parity))
    (PACKET/(prefix+'_GRAPH_SIGNATURES.json')).write_text(json.dumps(dict(rows=signatures),separators=(',',':'))+'\n')
    write(PACKET/(prefix+'_CORRECTNESS.json'),dict(status='PASS' if all(g['counters_and_packets_equal'] for g in gates.values()) else 'FAIL',rows=correctness))
    integrity=all(g['counters_and_packets_equal'] and all(x['all_h0_kernel_counts_preserved'] for x in g['kernel_identity']) for g in gates.values())
    result=dict(candidate=candidate,diagnostic_integrity_pass=integrity,status='DIAGNOSTIC_PASS' if all(g['diagnostic_gate'] for g in gates.values()) else 'DIAGNOSTIC_GATE_FAIL',rows=summaries,gates=gates,
                units='ms/step; host and GPU spans overlap. Expert/NCCL residency mean across ranks; arrival spreads max-min across ranks.',
                primary_timing=False,graph_service_note='Host launch ranges include scratch copies and replay. GPU expert kernel union excludes DMA; graph scratch D2D DMA is reported separately. Neither sum is primary TPOT.',sources=sources)
    write(PACKET/(prefix+'_DIAGNOSTIC_RESULTS.json'),result)
    with (PACKET/(prefix+'_DIAGNOSTIC_RESULTS.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    return result
if __name__=='__main__':
    import sys
    print(json.dumps(report(candidate='H1b' if '--wrapper' in sys.argv else 'H1')['gates'],indent=2))
