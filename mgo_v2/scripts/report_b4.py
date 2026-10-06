"""B4 mechanism summary, preserving service/readiness/host distinctions."""
import json,csv,sqlite3,bisect
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from report_b2 import export
from prepare_critical_microbench import ROOT,PACKET,write,sha

def correlation(x,y):
 v=float(spearmanr(x,y).statistic)
 return v if np.isfinite(v) else None

def dispatch_counts(cap,rank):
 db=sqlite3.connect((cap/f'interval_analysis/rank{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 strings=dict(db.execute('select id,value from StringIds'));ranges=defaultdict(list)
 for r in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end>start'):
  name=strings.get(r['textId'],r['text']) or ''
  if name.startswith('b2|') and name.endswith('|expert_loop'):ranges[r['globalTid']].append((r['start'],r['end'],int(name.split('|')[1])))
 for tid in ranges:ranges[tid].sort()
 starts={tid:[a for a,b,e in rs] for tid,rs in ranges.items()};counts=defaultdict(int);all_calls=defaultdict(int)
 candidates=defaultdict(list);apis=defaultdict(list)
 tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
 for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
  if table not in tables:continue
  for r in db.execute(f'select start,end,globalTid,nameId from {table}'):
   tid=r['globalTid']
   if tid not in ranges:continue
   i=bisect.bisect_right(starts[tid],r['start'])-1
   if i<0 or ranges[tid][i][1]<r['end']:continue
   e=ranges[tid][i][2];name=strings[r['nameId']];interval=(r['start'],r['end'])
   apis[e].append(interval)
   if 'LaunchKernel' in name or 'GraphLaunch' in name:candidates[e].append(interval)
 def outer_count(intervals):
  # Count direct Triton driver calls, but not the driver call nested inside
  # a torch CUDA-runtime launch as a second host dispatch.
  end=-1;count=0
  for a,b in sorted(intervals,key=lambda x:(x[0],-x[1])):
   if b<=end:continue
   count+=1;end=b
  return count
 for e,intervals in candidates.items():counts[e]=outer_count(intervals)
 for e,intervals in apis.items():all_calls[e]=outer_count(intervals)
 db.close();return counts,all_calls

def main():
 state=json.loads((ROOT/'b4/C30/diagnostic_status.json').read_text());assert state['status']=='DIAGNOSTICS_COMPLETE'
 calibration=json.loads((PACKET/'B4_GROUPED_SERVICE_CALIBRATION.json').read_text());co=np.array(calibration['coefficients'])
 def predict(v):return float(co@np.array([1,len(v),sum(v),sum((n+31)//32 for n in v),len(v)*((max(v)+31)//32)]))
 rows=[];waves=[];sources={};summaries=[];all_caps={}
 for path in state['completed']:
  cap=Path(path);case=json.loads((cap/'case.json').read_text());mode=case['b4_executor'];policy=case['policy'];all_caps[policy,mode]=cap
  ranks=export(cap)
  for rank,rdata in enumerate(ranks):
   calls,all_calls=dispatch_counts(cap,rank)
   if mode=='H2':evwaves={e['event']:e['waves'] for e in json.loads((cap/f'b4_waves_rank{rank}.json').read_text())['events']}
   for e in rdata['events']:
    phases=e['phases'];loop=phases['expert_loop'];event=e['event']
    if mode=='H2':vectors=[w['rows'] for w in evwaves[event]]
    else:vectors=[[d['rows']] for d in e['expert_calls']]
    pred=sum(predict(v) for v in vectors);nrows=sum(map(sum,vectors))
    for j,v in enumerate(vectors):waves.append(dict(policy=policy,executor=mode,rank=rank,event=event,wave=j,experts=len(v),rows=v,predicted_service_ms=predict(v)))
    wait=phases.get('expert_ready_wait',{}).get('host_union_ms',0.)
    rows.append(dict(policy=policy,executor=mode,rank=rank,event=event,expert_rows=nrows,experts=sum(map(len,vectors)),waves=len(vectors),pred_group_service_ms=pred,host_loop_ms=loop['host_union_ms'],observed_expert_stage_ms=e['expert_exclusive_account']['full_loop_span_ms'],grouped_gpu_active_ms=phases['expert_compiled_kernel']['gpu_union_ms'],ready_wait_ms=wait,host_dispatch_calls=calls[event],all_expert_cuda_api_calls=all_calls[event],forward_host_entry_ns=phases['forward_collective_host_call']['host_start_ns'],forward_gpu_start_ns=phases['forward_collective_host_call']['gpu_start_ns'],forward_gpu_end_ns=phases['forward_collective_host_call']['gpu_end_ns'],return_host_entry_ns=phases['return_collective_host_call']['host_start_ns'],return_gpu_start_ns=phases['return_collective_host_call']['gpu_start_ns'],forward_nccl_ms=phases['forward_collective_host_call']['gpu_union_ms'],return_nccl_ms=phases['return_collective_host_call']['gpu_union_ms']))
   for name in (f'rank{rank}.json',f'b2_host_rank{rank}.json',f'b2_rank{rank}_analysis.json',f'communication_rank{rank}.json',f'copy_trace_rank{rank}.json'):sources[str(cap/name)]=sha(cap/name)
  data=[r for r in rows if r['policy']==policy and r['executor']==mode]
  summary=dict(policy=policy,executor=mode)
  for name in ('host_loop_ms','observed_expert_stage_ms','grouped_gpu_active_ms','ready_wait_ms','host_dispatch_calls','all_expert_cuda_api_calls','pred_group_service_ms','forward_nccl_ms','return_nccl_ms'):summary[name+'_per_step']=sum(r[name] for r in data)/32
  summary['service_stage_spearman']=correlation([r['pred_group_service_ms'] for r in data],[r['observed_expert_stage_ms'] for r in data])
  summary['raw_rows_stage_spearman']=correlation([r['expert_rows'] for r in data],[r['observed_expert_stage_ms'] for r in data])
  summary['service_gpu_spearman']=correlation([r['pred_group_service_ms'] for r in data],[r['grouped_gpu_active_ms'] for r in data])
  summary['service_stage_aggregate_relative_error']=abs(sum(r['pred_group_service_ms']-r['observed_expert_stage_ms'] for r in data))/sum(r['observed_expert_stage_ms'] for r in data)
  predicted=[];observed=[];skews=[]
  for event in range(48,432):
   rs=[r for r in data if r['event']==event];assert len(rs)==4
   # Explicit service+readiness hypothesis, never fit to return timing.
   ready=[r['forward_gpu_end_ns']/1e6+r['pred_group_service_ms']+r['ready_wait_ms'] for r in rs]
   entry=[r['return_host_entry_ns']/1e6 for r in rs]
   predicted.append(float(np.ptp(ready)));observed.append(float(np.ptp(entry)))
   skews.append(dict(event=event,forward_host_ms=float(np.ptp([r['forward_host_entry_ns'] for r in rs])/1e6),forward_gpu_ms=float(np.ptp([r['forward_gpu_start_ns'] for r in rs])/1e6),return_host_ms=float(np.ptp(entry)),return_gpu_ms=float(np.ptp([r['return_gpu_start_ns'] for r in rs])/1e6),predicted_ready_ms=predicted[-1]))
  summary['ready_skew_return_entry_spearman']=correlation(predicted,observed);summary['arrival_skews']=skews;summaries.append(summary)
 gates={};parity=True
 for policy in ('BR','FCA'):
  a=next(r for r in summaries if r['policy']==policy and r['executor']=='H1b');b=next(r for r in summaries if r['policy']==policy and r['executor']=='H2')
  same=True
  for rank in range(4):
   ac=all_caps[policy,'H1b'];bc=all_caps[policy,'H2'];ar=json.loads((ac/f'rank{rank}.json').read_text());br=json.loads((bc/f'rank{rank}.json').read_text())
   for field in ('controller_counters','decode_expert_copies','decode_expert_bytes','transport_calls'):same &= ar[field]==br[field]
   same &= json.loads((ac/f'communication_rank{rank}.json').read_text())['events']==json.loads((bc/f'communication_rank{rank}.json').read_text())['events']
  parity &= same
  gates[policy]=dict(parity=bool(same),host_call_reduction=1-b['host_dispatch_calls_per_step']/a['host_dispatch_calls_per_step'],host_loop_reduction=1-b['host_loop_ms_per_step']/a['host_loop_ms_per_step'],service_stage_spearman=b['service_stage_spearman'])
 h2=[r for r in summaries if r['executor']=='H2'];br=next(r for r in h2 if r['policy']=='BR');fca=next(r for r in h2 if r['policy']=='FCA')
 predicted=fca['pred_group_service_ms_per_step']-br['pred_group_service_ms_per_step'];actual=fca['observed_expert_stage_ms_per_step']-br['observed_expert_stage_ms_per_step'];waitdelta=fca['ready_wait_ms_per_step']-br['ready_wait_ms_per_step']
 gap=dict(status='DIAGNOSTIC_COMPLETE',summaries=summaries,gates=gates,delta=dict(predicted_ms_per_step=predicted,observed_ms_per_step=actual,relative_error=abs(predicted-actual)/max(abs(actual),1e-9),measured_ready_wait_delta_ms_per_step=waitdelta,residual_after_readiness_ms_per_step=actual-predicted-waitdelta),exception_applied=False,notes=['No arbitrary scale or coefficient fit to policy timing.','H1b service predictor is summed singleton grouped service, a diagnostic comparison only.','Ready-wait is separately measured host time, not automatically an additive physical critical-path correction.','Launch count is main-thread CUDA runtime/driver kernel/graph dispatch calls within expert-loop, excluding event queries and deduplicating nested driver calls. All top-level CUDA API calls also reported.'])
 write(PACKET/'B4_KNOB_RUNTIME_GAP.json',gap)
 for name,data in [('B4_WAVE_STATS',waves),('B4_DIAGNOSTIC_RESULTS',rows)]:
  write(PACKET/(name+'.json'),dict(status='PASS' if parity else 'FAIL',primary_timing=False,rows=data,sources=sources))
  with (PACKET/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 assert parity,'H1b/H2 diagnostic counters mismatch'
if __name__=='__main__':main()
