"""Audit one-cell paired diagnostic, retaining every rank and layer."""
import csv,gzip,hashlib,json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/br_near_phase_probe_20261007');PACKET=P/'experiments/r4_br_near_h0_20261007/diagnostic'
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 assert json.loads((ROOT/'status.json').read_text())['status']=='PASS'
 policies={};rank_rows=[];event_rows=[]
 for policy in ['BR','LA_CA_NEAR']:
  ranks=[];events=[]
  for rank in range(4):
   receipt=json.loads((ROOT/f'{policy}_diagnostic_rank{rank}.json').read_text());assert receipt['status']=='PASS' and receipt['tokens_match_primary'] and receipt['no_compile']
   with gzip.open(ROOT/f'{policy}_trace_rank{rank}.json.gz','rt') as f:raw=json.load(f)
   byevent={i:defaultdict(float) for i in range(48,1584)}
   for s in raw['spans']:
    assert s['event'] in byevent and s['ms']>=0
    byevent[s['event']][s['name']]+=s['ms']
   for c in raw['copies']:byevent[c['event']]['dma']+=c['ms'];byevent[c['event']]['copies']+=1
   for c in raw['cpu']:byevent[c['event']][c['name']]+=c['ms']
   for w in raw['work']:byevent[w['event']].update({k:v for k,v in w.items() if k!='event'})
   for event,d in byevent.items():
    d['expert_service']=d['compute']-d['h2d_wait'];d['payload_comm']=d['forward']+d['return'];d['moe_other']=d['moe']-d['compute']-d['metadata']-d['payload_comm']
    assert d['expert_service']>=-.03 and d['moe_other']>=-.03,(policy,rank,event,dict(d))
    row=dict(policy=policy,rank=rank,event=event,step=event//48,layer=event%48,**d);event_rows.append(row)
   total={k:sum(d[k] for d in byevent.values())/32 for k in next(iter(byevent.values()))}
   total.update(policy=policy,rank=rank,physical_gpu=receipt['physical_gpu'],diagnostic_tpot_ms=receipt['TPOT']*1000,primary_tpot_ms=receipt['primary_TPOT']*1000)
   total['forward_MB_per_step']=receipt['forward_bytes']/1e6/32;total['return_MB_per_step']=receipt['return_bytes']/1e6/32
   total['outside_moe']=total['diagnostic_tpot_ms']-total['moe'];assert total['outside_moe']>=-.1
   total['partition_sum']=sum(total[k] for k in ['expert_service','h2d_wait','payload_comm','metadata','moe_other','outside_moe'])
   assert abs(total['partition_sum']-total['diagnostic_tpot_ms'])<.01
   ranks.append(total);rank_rows.append(total);events.append(byevent)
  critical=max(ranks,key=lambda x:x['diagnostic_tpot_ms'])
  metrics=['expert_rows','expert_service','compute','h2d_wait','payload_comm','dma','plan_current','plan_prefetch_next','copies','forward_remote_packets']
  per_layer={}
  for metric in metrics:
   values=np.array([[e[i][metric] for e in events] for i in range(48,1584)])
   per_layer[metric]=dict(mean_rank_per_step=float(values.mean(axis=1).sum()/32),max_rank_per_layer_per_step=float(values.max(axis=1).sum()/32),max_minus_mean_per_step=float((values.max(axis=1)-values.mean(axis=1)).sum()/32))
  policies[policy]=dict(ranks=ranks,critical_rank=critical['rank'],critical_partition=critical,rank_imbalance=per_layer,global_diagnostic_tpot_ms=max(x['diagnostic_tpot_ms'] for x in ranks),global_primary_tpot_ms=max(x['primary_tpot_ms'] for x in ranks))
 br=policies['BR'];near=policies['LA_CA_NEAR']
 summary=dict(status='PASS',condition='R4 0,1,4,5 C30 local B64 input512 decode32 H0 full pinned V3 P2 T2',policies=policies,primary_gain=1-near['global_primary_tpot_ms']/br['global_primary_tpot_ms'],diagnostic_gain=1-near['global_diagnostic_tpot_ms']/br['global_diagnostic_tpot_ms'],critical_partition_br_minus_near={k:br['critical_partition'][k]-near['critical_partition'][k] for k in ['expert_service','h2d_wait','payload_comm','metadata','moe_other','outside_moe']},limits=['Single diagnostic per policy; no causal intervention or repeat-stability proof.','GPU event spans include host launch gaps and instrumentation.','Collective completion includes peer arrival wait, not pure wire time.','DMA overlaps execution; do not add DMA or rank imbalance to TPOT partition.','Original primary timings retained separately, no scaling diagnostic spans onto primary.'])
 write(PACKET/'SUMMARY.json',summary)
 for name,rows in [('RANK_COMPONENTS.csv',rank_rows),('LAYER_COMPONENTS.csv',event_rows)]:
  keys=list(dict.fromkeys(k for row in rows for k in row))
  with (PACKET/name).open('w') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 dest=PACKET/'raw';dest.mkdir(exist_ok=True);hashes={}
 for f in sorted(ROOT.iterdir()):
  if f.is_file():shutil.copy2(f,dest/f.name);hashes[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
 write(PACKET/'SOURCE_HASHES.json',hashes)
 print(json.dumps({k:v for k,v in summary.items() if k!='policies'},indent=2))
 for policy,p in policies.items():print(policy,json.dumps(p['critical_partition'],indent=2),json.dumps(p['rank_imbalance'],indent=2))
if __name__=='__main__':main()
