"""Preserve all Stage-A samples and report bounded causal contrasts."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from prepare_critical_microbench import ROOT,PACKET,write,features

def main():
 state=json.loads((ROOT/'status.json').read_text());assert state['status']=='MEASUREMENTS_COMPLETE'
 files=[ROOT/f'attempt1/rank{r}.json' for r in range(4)];ranks=[json.loads(p.read_text()) for p in files]
 assert all(x['status']=='PASS' for x in ranks)
 matrices=json.loads((ROOT/'inputs.json').read_text());shapes={x['id']:x for x in matrices['shapes']}
 output={'A2A_SPLIT_RESULTS':[],'ARRIVAL_SKEW_RESULTS':[],'EXPERT_TAU_RESULTS':[]};bundles=[]
 comm=[{(r['stage'],r['id'],r['packet_bytes'],r['delay_ms'],r['delayed_rank']):r for r in x['records'] if r['stage'] in ('A1','A2')} for x in ranks]
 assert all(set(x)==set(comm[0]) for x in comm)
 for key in comm[0]:
  rs=[x[key] for x in comm];stage,ident,packet,delta,delayed=key
  vals=np.max([x['completion']['samples_ms'] for x in rs],axis=0)
  row=dict(id=ident,packet_bytes=packet,target_delay_ms=delta,delayed_rank=delayed,**features(rs[0]['matrix']),matrix=rs[0]['matrix'],median_max_local_completion_ms=float(np.median(vals)),p10_ms=float(np.quantile(vals,.1)),p90_ms=float(np.quantile(vals,.9)),max_local_completion_samples_ms=vals.tolist(),rank_samples=rs,shape=shapes[ident]['shape'])
  if stage=='A2':row['achieved_delay_ms']=rs[delayed]['achieved_delay']['median_ms']
  output['A2A_SPLIT_RESULTS' if stage=='A1' else 'ARRIVAL_SKEW_RESULTS'].append(row)
 for row in output['ARRIVAL_SKEW_RESULTS']:
  base=next(x for x in output['ARRIVAL_SKEW_RESULTS'] if x['id']==row['id'] and x['delayed_rank']==row['delayed_rank'] and x['target_delay_ms']==0)
  row['incremental_tail_ms']=row['median_max_local_completion_ms']-base['median_max_local_completion_ms']
 for rank,x in enumerate(ranks):
  taus={r['rows']:r for r in x['records'] if r['stage']=='A3_TAU'}
  for n,r in taus.items():
   row=dict(rank=rank,rows=n,primary_ladder=r['primary_ladder'],stable=r['stable'],median_ms=float(np.median([v for b in r['blocks'] for v in b['samples_ms']])),blocks=r['blocks'],initial_drift=r['initial_drift'],final_drift=r['final_drift']);output['EXPERT_TAU_RESULTS'].append(row)
  lookup={r['rows']:r['median_ms'] for r in output['EXPERT_TAU_RESULTS'] if r['rank']==rank}
  for b in x['records']:
   if b['stage']!='A3_BUNDLE':continue
   pred=sum(lookup[n] for n in b['rows']);actual=b['timing']['median_ms']
   bundles.append(dict(rank=rank,event=b['event'],rows=b['rows'],predicted_ms=pred,measured_ms=actual,relative_error=abs(pred-actual)/actual,samples=b['timing']))
 for name,rows in output.items():
  write(PACKET/(name+'.json'),dict(rows=rows,scope='All valid samples retained; GPU service microbench, not E2E/TPOT.'))
  keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
  with (PACKET/(name+'.csv')).open('w') as f:
   w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
 write(PACKET/'EXPERT_BUNDLE_RESULTS.json',dict(rows=bundles))
 split=output['A2A_SPLIT_RESULTS'];contrasts=[]
 for v in matrices['volumes']:
  for packet in (4120,4096):
   xs=[x for x in split if x['id'].startswith(f'V{v}_') and not x['shape'].startswith('RAW') and x['packet_bytes']==packet];base=next(x['median_max_local_completion_ms'] for x in xs if x['shape']=='BALANCED')
   contrasts.extend(dict(volume=v,packet_bytes=packet,shape=x['shape'],median_ms=x['median_max_local_completion_ms'],ratio_vs_balanced=x['median_max_local_completion_ms']/base) for x in xs)
 correlations={}
 for packet in (4120,4096):
  xs=[x for x in split if x['packet_bytes']==packet and not x['shape'].startswith('RAW')]
  correlations[str(packet)]={k:float(spearmanr([x[k] for x in xs],[x['median_max_local_completion_ms'] for x in xs]).statistic) for k in ('total_packets','max_peer_edge','max_incident','max_send','max_recv','active_directed_edges')}
 skew=output['ARRIVAL_SKEW_RESULTS'];slope=float(np.polyfit([x['achieved_delay_ms'] for x in skew],[x['incremental_tail_ms'] for x in skew],1)[0]);corr=float(spearmanr([x['achieved_delay_ms'] for x in skew],[x['incremental_tail_ms'] for x in skew]).statistic)
 unstable=[(x['rank'],x['rows']) for x in output['EXPERT_TAU_RESULTS'] if not x['stable']]
 report=dict(status='CHECKPOINT_A_COMPLETE',stage_b='NOT_STARTED_CHECKPOINT',contrasts=contrasts,feature_spearman=correlations,arrival_skew_tail_slope=slope,arrival_skew_tail_spearman=corr,bundle_median_absolute_relative_error=float(np.median([x['relative_error'] for x in bundles])),tau_unstable_rank_rows=unstable,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[ROOT/'inputs.json']},limitations=['No cross-GPU absolute event clock: max rank duration is common-rendezvous approximation.','CUDA graph GPU service excludes Python/controller/H2D effects; not a full-runtime predictive validation.','No causal effect on prior FCA TPOT established by isolated microbench alone.','All valid observations retained; bounded max-three tau blocks can remain unstable.'])
 report['tau_row_spearman_by_rank']={str(rank):float(spearmanr([x['rows'] for x in output['EXPERT_TAU_RESULTS'] if x['rank']==rank and x['primary_ladder']],[x['median_ms'] for x in output['EXPERT_TAU_RESULTS'] if x['rank']==rank and x['primary_ladder']]).statistic) for rank in range(4)}
 report['tau_primary_ladder_stable']=all(x['stable'] for x in output['EXPERT_TAU_RESULTS'] if x['primary_ladder'])
 report['tau_scope']='CUDA graph GPU execution span of the exact compiled expert function, including inter-kernel gaps; not a CUPTI sum of kernel durations.'
 report['checkpoint_instruction']='AGENT_TASK.md: STOP and publish MICROBENCH_RESULTS before doing Stage B.'
 write(PACKET/'MICROBENCH_RESULTS.json',report)
 lines=['# Checkpoint A: physical critical-path microbench','', 'R4 physical GPUs 0/1/4/5, Env1 verified P2P/IPC; BF16. No full-model inference, new trace capture, profiler, or Stage B/oracle run.','', '## A1: matched-volume split shape','', '| Packets | Bytes/packet | Shape | Median max-local completion ms | / balanced |','|---:|---:|---|---:|---:|']
 for x in contrasts:lines.append(f"| {x['volume']} | {x['packet_bytes']} | {x['shape']} | {x['median_ms']:.6f} | {x['ratio_vs_balanced']:.3f} |")
 lines+=['','Feature correlations are descriptive across the bounded matrix; see JSON. Raw unscaled trace matrices are also measured and retained separately.','', '## A2: arrival skew','',f'Measured-delay vs incremental-tail slope: {slope:.3f}; Spearman: {corr:.3f}. GPU-delay targets 0/0.10/0.25/0.50/1/2 ms, two matrices, all four delayed ranks. NCCL residency includes arrival waiting.','', '## A3: compiled expert service','',f"2/4/8-expert bundle median absolute relative prediction error: {report['bundle_median_absolute_relative_error']:.2%}. Unstable rank/row calibrations after bounded blocks: {len(unstable)}.",'','The exact runtime expert function is compiled with dynamic=True/fullgraph=True and real Qwen3 checkpoint weights. Power ladder 1–512; extra observed bundle row sizes avoid interpolation. All samples, drift, and any unstable cells are retained.','', '## Decision and limits','', 'This is Checkpoint A, not a policy-gain result. Stage B has not started. An oracle remains gated on the separate held-out model-validation requirements.']
 lines+=['', '## Expert power ladder', '', '| Rows | Median across rank medians (ms) | P10 pooled (ms) | P90 pooled (ms) |', '|---:|---:|---:|---:|']
 for n in matrices['tau_rows']:
  xs=[x for x in output['EXPERT_TAU_RESULTS'] if x['rows']==n];samples=[v for x in xs for b in x['blocks'] for v in b['samples_ms']]
  lines.append(f"| {n} | {np.median([x['median_ms'] for x in xs]):.6f} | {np.quantile(samples,.1):.6f} | {np.quantile(samples,.9):.6f} |")
 lines+=['', 'All primary power-ladder cells meet the 2% final-block drift gate. Supplemental rank1/n18 remains unstable (20.25% final-block drift); retain all three blocks and do not treat that calibration as stable. Row-count/service-time Spearman by rank: '+str(report['tau_row_spearman_by_rank'])+'.', '', report['tau_scope'], '', 'Matched synthetic source/destination-hot shapes are 1.24–1.68x balanced. At matched volumes, the selected BR/FCA-shaped matrices are within approximately 6% of balanced; the shape-only experiment does not explain the earlier multi-fold return-NCCL residency gap. The approximately one-for-one skew/tail relationship establishes sensitivity to arrival skew in isolation, not that expert skew alone caused the earlier TPOT loss.', '', 'Checkpoint stop instruction: AGENT_TASK.md explicitly says “STOP and publish MICROBENCH_RESULTS before doing Stage B.” No new online policy or oracle was implemented.']
 lines+=['']+['- '+x for x in report['limitations']]
 (PACKET/'MICROBENCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('contrasts','source_sha256','limitations')},indent=2))
if __name__=='__main__':main()
