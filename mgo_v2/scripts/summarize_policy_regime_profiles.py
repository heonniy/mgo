"""Reconcile physical packet traffic and separate rank-local GPU/host phases."""
import json,statistics,subprocess,hashlib
from pathlib import Path
from run_refactor_measure import PACKET
from refactor_h2d_steps import summarize_rank
import run_timing_stability as h
from policy_regime_paths import ROOT,PACKET,ENVIRONMENT
def distribution(values):
 return dict(mean=statistics.mean(values),min=min(values),max=max(values),max_over_mean=max(values)/statistics.mean(values) if statistics.mean(values) else None)
def main():
 state=json.loads((ROOT/'profile_status.json').read_text());assert state['status']=='CAPTURES_COMPLETE'
 rows=[];sources={}
 def read(p):
  raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
 for path in state['completed']:
  capture=Path(path);status=read(capture/'status.json');assert status['status']=='PASS' and status['world']==4
  analysis=capture/'interval_analysis'
  if not (analysis/'summary.json').exists():
   assert not analysis.exists(),'repair incomplete export into a new directory explicitly'
   with (capture/'export_driver.log').open('w') as log:subprocess.run([h.PYTHON,str(h.P/'scripts/export_refactor_profiles.py'),str(capture),'--nsys',status['case'].get('profile_nsys_binary','/usr/local/bin/nsys')],stdout=log,stderr=subprocess.STDOUT,check=True)
  data=[read(analysis/f'rank{r}_intervals.json') for r in range(4)]
  comm=[read(capture/f'communication_rank{r}.json') for r in range(4)]
  phases=('moe.forward_a2a.nccl','moe.return_a2a.nccl','moe.expert_compute')
  metrics={phase:distribution([row['kernel_union_ms'][phase]/64 for row in data]) for phase in phases}
  for name in ('moe.current_controller','moe.prefetch_controller','moe.host_staging'):
   assert all(name in row['cpu_nvtx'] for row in data),name
   metrics['CPU '+name]=distribution([row['cpu_nvtx'][name]['total_ms']/64 for row in data])
  dma=[summarize_rank(row,64) for row in data]
  for key in dma[0]['mean_per_step']:metrics[key]=distribution([row['mean_per_step'][key] for row in dma])
  traffic=[]
  for i in range(3072):
   event=i+48;rs=[c['events'][i] for c in comm];assert all(x['event']==event for x in rs)
   for src in range(4):
    for dst in range(4):assert rs[src]['send_token_rows'][dst]==rs[dst]['recv_token_rows'][src]
   sent=[sum(x['send_token_rows'])-x['send_token_rows'][r] for r,x in enumerate(rs)]
   recv=[sum(x['recv_token_rows'])-x['recv_token_rows'][r] for r,x in enumerate(rs)]
   assert sum(sent)==sum(recv)
   traffic.append(dict(event=event,remote_token_rank_pairs=sum(sent),max_rank_incident_packets=max(a+b for a,b in zip(sent,recv)),expert_rows=distribution([x['expert_rows'] for x in rs]),mandatory_fetches=distribution([x['mandatory_fetches'] for x in rs])))
  imbalance={}
  for phase in phases:
   values=[]
   for i in range(48,3120):
    xs=[x['interval_decomposition']['causal_event_kernel_union_ms'][str(i)].get(phase,0.) for x in data]
    values.append(distribution(xs))
   imbalance[phase]=dict(mean_event_max_over_mean=statistics.mean(v['max_over_mean'] for v in values if v['max_over_mean'] is not None),sum_event_max_ms_per_step=sum(v['max'] for v in values)/64,scope='Per-event maximum is an imbalance proxy, not a sum defining wall time.')
  rows.append(dict(setting=capture.parent.name,policy=status['case']['policy'],rank_local_ms_per_step=metrics,imbalance=imbalance,remote_token_rank_pairs=sum(x['remote_token_rank_pairs'] for x in traffic),sum_event_max_rank_incident_packets=sum(x['max_rank_incident_packets'] for x in traffic),traffic_per_event=traffic,capture=str(capture)))
 assert {(x['setting'],x['policy']) for x in rows}=={(s,p) for s in ('C30','C60') for p in ('BR','OLD_CA','FCA','LA_CA')}
 report=dict(status='PASS',rows=rows,source_sha256=sources,scope='Separate instrumented attribution; primary TPOT/E2E are in POLICY_REGIME_TIMING_RESULTS.json. H2D outside COMM/COMPUTE is an overlap definition, not proven stall. NCCL residency includes waiting. CPU NVTX times are host wall time, not GPU kernels. Rank distributions are never summed into step wall time. One capture per policy is descriptive, not repeated timing evidence.')
 (PACKET/'POLICY_REGIME_MECHANISMS.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS: eight captures, four ranks each; packet send/receive counts reconciled')
if __name__=='__main__':main()
