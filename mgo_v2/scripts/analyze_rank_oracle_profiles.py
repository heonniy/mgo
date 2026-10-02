#!/usr/bin/env python3
"""Stream Nsight activities into per-event/rank phase rows, without loading traces."""
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import re
import sqlite3
from pathlib import Path

from run_rank_oracle_study import ROOT, OUT, write


def duration(intervals):
    total=0; right=-1
    for a,b in sorted(intervals):
        total += max(0,b-max(a,right)); right=max(right,b)
    return total/1e6


def analyze(policy):
    name=f'profile_b8_{policy}'; root=ROOT/name
    db=sqlite3.connect(f'file:{root / "profile.sqlite"}?mode=ro',uri=True)
    db.execute('PRAGMA cache_size=-32768')
    raw=list(db.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'oracle:%'"))
    if not raw: raise RuntimeError('missing oracle NVTX scopes')
    windows=defaultdict(list); phases={}; records={}
    for a,b,tid,label in raw:
        _,rank,event,phase=label.split(':'); rank,event=int(rank),int(event)
        if b is None or b<a: raise RuntimeError('incomplete NVTX range')
        phases[rank,event,phase]=(a,b)
        if phase=='moe_layer':
            pid=tid & ~((1<<24)-1)
            windows[pid].append((a,b,rank,event))
            records[rank,event]=dict(policy=policy,rank=rank,event=event,step=event//48,layer=event%48,
                                    moe_cpu_ms=(b-a)/1e6,expert=[],gemm=[],h2d=[],allgpu=[],
                                    routing_metadata=[],dispatch=[],combine=[],nccl=[],gemm_kernel_count=0,
                                    h2d_bytes=0)
    starts={}
    for pid,items in windows.items():
        items.sort(); starts[pid]=[a for a,b,r,e in items]
        assert all(x[1]<=y[0] for x,y in zip(items,items[1:]))
    assert len(records)==4*65*48, len(records)
    def locate(pid,a,b):
        i=bisect_right(starts.get(pid,[]),a)-1
        if i<0: return None
        begin,end,rank,event=windows[pid][i]
        if b>end: return None
        return records[rank,event]
    # Associate asynchronous kernels through their CPU launch correlation,
    # rather than requiring device completion inside the CPU submission range.
    strings=dict(db.execute('SELECT id,value FROM StringIds'))
    launch_ids=[i for i,value in strings.items() if 'launch' in value.lower()]
    nccl_ids={i for i,value in strings.items() if 'nccl' in value.lower()}
    gemm_ids={i for i,value in strings.items() if re.search(r'gemm|gemv|matmul|cutlass|^nvjet_tst_',value,re.I)}
    launches={}
    launch_sql='SELECT globalTid,start,end,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ('+','.join('?' for _ in launch_ids)+')'
    for tid,a,b,correlation in db.execute(launch_sql,launch_ids):
        pid=tid & ~((1<<24)-1)
        row=locate(pid,a,b)
        if row is None: continue
        phase=None
        for candidate in ('routing_metadata','dispatch','expert_execution','combine'):
            left,right=phases[row['rank'],row['event'],candidate]
            if left<=a and b<=right: phase=candidate; break
        launches[pid,correlation]=(row['rank'],row['event'],phase)
    kernel_names=defaultdict(int); expert_kernel_names=defaultdict(int)
    sql='SELECT globalPid,start,end,demangledName,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL'
    for pid,a,b,name_id,correlation in db.execute(sql):
        name_=strings[name_id]
        associated=launches.get((pid,correlation))
        if associated is None: continue
        rank,event,phase=associated
        row=records[rank,event]
        interval=(a,b); row['allgpu'].append(interval)
        key=(row['rank'],row['event'])
        if name_id in nccl_ids:
            row['nccl'].append(interval)
            if phase not in ('routing_metadata','dispatch','combine'):
                raise RuntimeError(f'unattributed NCCL launch: {key} {phase}')
            row[phase].append(interval)
        if phase=='expert_execution' and name_id not in nccl_ids:
            row['expert'].append(interval); expert_kernel_names[name_]+=1
            # This CUDA build emits native expert linears as nvjet_tst_*;
            # native source has three linears and per-event counts reconcile.
            if name_id in gemm_ids:
                row['gemm'].append(interval); row['gemm_kernel_count']+=1
                kernel_names[name_]+=1
    for pid,a,b,size in db.execute('SELECT globalPid,start,end,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1'):
        row=locate(pid,a,b)
        if row is None: continue
        row['allgpu'].append((a,b))
        if size>=9437184//3:
            row['h2d'].append((a,b)); row['h2d_bytes']+=size
    result=[]
    for rank in range(4):
        receipt=json.loads((root/'receipts'/f'{name}-rep0-rank{rank}.json').read_text())
        evidence=json.loads((root/'receipts'/f'{name}-evidence-rank{rank}.json').read_text())
        baseline=json.loads((ROOT/'timing_b8/receipts'/f'b8_{policy}_rep0-evidence-rank{rank}.json').read_text())
        assert evidence['events']==baseline['events'] and evidence['tokens']==baseline['tokens'], 'profile parity'
        assert sum(row['h2d_bytes'] for (r,e),row in records.items() if r==rank)==receipt['host_fetch_bytes'], 'H2D accounting'
        for e in evidence['events']:
            row=records[rank,e['event']]
            row['expert_rows']=e['loads'][rank]; row['distinct_experts']=e['distinct'][rank]
            if row['gemm_kernel_count']<row['distinct_experts']*3:
                write(root/'attribution_debug.json',dict(rank=rank,event=e['event'],expected=row['distinct_experts']*3,actual=row['gemm_kernel_count'],expert_kernel_names=dict(expert_kernel_names),gemm_kernel_names=dict(kernel_names)))
                raise AssertionError(f'missing expert GEMMs rank={rank} event={e["event"]} expected={row["distinct_experts"]*3} actual={row["gemm_kernel_count"]}')
            for category in ('expert','gemm','h2d','allgpu','routing_metadata','dispatch','combine','nccl'):
                values=row.pop(category)
                row[category+'_gpu_union_ms']=duration(values)
                row[category+'_gpu_span_ms']=(max(b for a,b in values)-min(a for a,b in values))/1e6 if values else 0.
                row[category+'_gpu_sum_ms']=sum(b-a for a,b in values)/1e6
            result.append(row)
    write(OUT/f'profile_{policy}_kernel_names.json',dict(kernel_names))
    write(OUT/f'profile_{policy}_kernel_validation.json',dict(status='PASS',
        classified_gemm_kernels=sum(r['gemm_kernel_count'] for r in result),
        expected_native_linear_calls=3*sum(r['distinct_experts'] for r in result),
        exact_three_per_expert=all(r['gemm_kernel_count']==3*r['distinct_experts'] for r in result),
        classification='Native expert ranges only; gemm/gemv/matmul/cutlass or observed nvjet_tst_* kernels.'))
    return result


def main():
    rows=[]
    for policy in ('P0','P1','O0'): rows.extend(analyze(policy))
    (OUT/'gpu_phase_profile.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    with (OUT/'gpu_phase_profile.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    grouped=defaultdict(list)
    for row in rows: grouped[row['policy'],row['event']].append(row)
    critical=[]
    for (policy,event),ranks in sorted(grouped.items()):
        critical.append(dict(policy=policy,event=event,step=event//48,layer=event%48,
            max_rank_expert_rows=max(r['expert_rows'] for r in ranks),
            **{'max_rank_'+metric:max(r[metric] for r in ranks) for metric in
               ('expert_gpu_union_ms','expert_gpu_span_ms','gemm_gpu_union_ms','nccl_gpu_union_ms',
                'h2d_gpu_union_ms','moe_cpu_ms','allgpu_gpu_span_ms')}))
    (OUT/'gpu_critical_path.json').write_text(json.dumps(critical,separators=(',',':'))+'\n')
    with (OUT/'gpu_critical_path.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(critical[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(critical)

if __name__=='__main__': main()
