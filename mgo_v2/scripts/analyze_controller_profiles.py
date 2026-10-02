#!/usr/bin/env python3
"""Short posthoc P1 profile: unchanged expert work and explicit planner transport."""
from bisect import bisect_right
from collections import defaultdict
import csv
import json
import re
import sqlite3
from pathlib import Path
from analyze_rank_oracle_profiles import duration
from run_controller_overhead import ROOT,OUT,ORIGINAL


def main():
    root=ROOT/'profile_P1'
    db=sqlite3.connect(f'file:{root/"profile.sqlite"}?mode=ro',uri=True);db.execute('PRAGMA cache_size=-32768')
    phases={};windows=defaultdict(list);records={}
    cells={c['name']:c['controller'] for c in json.loads((root/'cells.json').read_text())}
    payload_sizes={cell:json.loads((root/f'receipts/{cell}-controller-rank0.json').read_text())['payload_bytes_per_event'] for cell in cells}
    for a,b,tid,text in db.execute("SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'controller:%'"):
        _,cell,rank,event,phase=text.split(':');rank,event=int(rank),int(event)
        assert b is not None and b>=a
        key=cell,rank,event;phases[key,phase]=(a,b)
        if phase=='moe_layer':
            pid=tid&~((1<<24)-1);windows[pid].append((a,b,key))
            records[key]=dict(controller=cells[cell],policy='P1',rank=rank,event=event,step=event//48,layer=event%48,
                moe_cpu_ms=(b-a)/1e6,expert=[],gemm=[],h2d=[],nccl=[],planner_broadcast=[],
                routing_metadata=[],dispatch=[],combine=[],gemm_kernel_count=0,h2d_bytes=0,
                planner_h2d_bytes=0,planner_d2h_bytes=0,planner_payload_h2d_bytes=0,planner_payload_d2h_bytes=0)
    assert len(records)==2*4*9*48,len(records)
    starts={}
    for pid,w in windows.items():w.sort();starts[pid]=[v[0] for v in w]
    def locate(pid,a,b):
        i=bisect_right(starts.get(pid,[]),a)-1
        if i<0:return None
        begin,end,key=windows[pid][i]
        return key if b<=end else None
    launches={}
    for tid,a,b,correlation in db.execute("SELECT r.globalTid,r.start,r.end,r.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE s.value LIKE '%Launch%' OR s.value LIKE '%Memcpy%'"):
        pid=tid&~((1<<24)-1);key=locate(pid,a,b)
        if key is None:continue
        phase=None
        for name in ('planner_broadcast','routing_metadata','dispatch','expert_execution','combine'):
            if (key,name) in phases:
                left,right=phases[key,name]
                if left<=a and b<=right:phase=name;break
        launches[pid,correlation]=(key,phase)
    for pid,a,b,name,correlation in db.execute('SELECT k.globalPid,k.start,k.end,s.value,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id'):
        association=launches.get((pid,correlation))
        if association is None:continue
        key,phase=association;row=records[key]
        if 'nccl' in name.lower():
            assert phase in ('planner_broadcast','routing_metadata','dispatch','combine'),(key,phase)
            row['nccl'].append((a,b));row[phase].append((a,b))
        elif phase=='expert_execution':
            row['expert'].append((a,b))
            if re.search(r'gemm|gemv|matmul|cutlass|^nvjet_tst_',name,re.I):
                row['gemm'].append((a,b));row['gemm_kernel_count']+=1
    for pid,a,b,size,kind,correlation in db.execute('SELECT globalPid,start,end,bytes,copyKind,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
        association=launches.get((pid,correlation))
        if association is not None and association[1]=='planner_broadcast':
            key,phase=association
            if kind==1:records[key]['planner_h2d_bytes']+=size
            elif kind==2:records[key]['planner_d2h_bytes']+=size
            if size==payload_sizes[key[0]]:
                if kind==1:records[key]['planner_payload_h2d_bytes']+=size
                elif kind==2:records[key]['planner_payload_d2h_bytes']+=size
        if kind==1 and size>=9437184//3:
            key=locate(pid,a,b)
            if key is not None:
                records[key]['h2d'].append((a,b));records[key]['h2d_bytes']+=size
    original={}
    with (ORIGINAL/'gpu_phase_profile.csv').open() as f:
        for row in csv.DictReader(f):
            if row['policy']=='P1' and int(row['step'])<9:
                original[int(row['rank']),int(row['event'])]=row
    rows=[];checks=[]
    for cell in cells:
        for rank in range(4):
            receipt=json.loads((root/f'receipts/{cell}-rep0-rank{rank}.json').read_text())
            evidence=json.loads((root/f'receipts/{cell}-evidence-rank{rank}.json').read_text())
            meta=json.loads((root/f'receipts/{cell}-controller-rank{rank}.json').read_text())
            assert evidence['status']=='PASS' and len(evidence['events'])==432
            selected=[records[cell,rank,event] for event in range(432)]
            assert sum(v['h2d_bytes'] for v in selected)==receipt['host_fetch_bytes']
            for e in evidence['events']:
                row=records[cell,rank,e['event']];base=original[rank,e['event']]
                row['expert_rows']=e['loads'][rank];row['distinct_experts']=e['distinct'][rank]
                assert row['gemm_kernel_count']>=3*row['distinct_experts']
                assert row['gemm_kernel_count']==int(base['gemm_kernel_count'])
                assert row['expert_rows']==int(base['expert_rows'])
                assert row['h2d_bytes']==int(base['h2d_bytes'])
                for category in ('expert','gemm','h2d','nccl','planner_broadcast','routing_metadata','dispatch','combine'):
                    intervals=row.pop(category);row[category+'_gpu_union_ms']=duration(intervals)
                rows.append(row)
            if cells[cell]=='C2':
                expected=meta['logical_payload_bytes']
                assert sum(v['planner_payload_h2d_bytes'] for v in selected)==(expected if rank==0 else 0)
                assert sum(v['planner_payload_d2h_bytes'] for v in selected)==(0 if rank==0 else expected)
            checks.append(dict(controller=cells[cell],rank=rank,events=432,status='PASS',
                               expert_rows_gemm_counts_fetch_bytes_equal_C0=True,transport_bytes_reconciled=True))
    with (OUT/'gpu_profile_events.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    totals=[]
    for variant in ('C0','C1','C2'):
        selected=list(original.values()) if variant=='C0' else [r for r in rows if r['controller']==variant]
        decoded=[r for r in selected if 0<int(r['step'])<9]
        row=dict(controller=variant,policy='P1',decode_forwards=8,ranks=4)
        for key in ('expert_gpu_union_ms','gemm_gpu_union_ms','nccl_gpu_union_ms','h2d_gpu_union_ms'):
            row['sum_event_max_rank_'+key]=sum(max(float(r[key]) for r in decoded if int(r['event'])==event) for event in range(48,432))
        row['planner_broadcast_gpu_union_sum_ms']=sum(float(r.get('planner_broadcast_gpu_union_ms',0)) for r in decoded)
        totals.append(row)
    (OUT/'gpu_profile_summary.json').write_text(json.dumps(totals,indent=2)+'\n')
    (OUT/'gpu_profile_validation.json').write_text(json.dumps(dict(status='PASS',scope='P1 prefill plus 8 decode forwards; C0 prefix from original profile',checks=checks),indent=2)+'\n')

if __name__=='__main__':main()
