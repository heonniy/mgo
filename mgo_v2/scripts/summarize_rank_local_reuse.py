#!/usr/bin/env python3
"""Compact CPU accounting summaries and a static B8 frontier figure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import resource
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
import csv,gzip,hashlib,json,math,statistics,time
from collections import Counter
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'experiments/rank_local_reuse_oracle_20261003'
OLD=P.parent/'fetch_comm_pareto_p2p_20261002'
ROOT=Path('/home/hwlee/mgo-results/rank_local_reuse_oracle_20261003')

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,x):(P/name).write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(name,rows):
 with (P/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def dominates(a,b):return a['peer_activation_bytes']<=b['peer_activation_bytes'] and a['expert_h2d_bytes']<=b['expert_h2d_bytes'] and (a['peer_activation_bytes'],a['expert_h2d_bytes'])!=(b['peer_activation_bytes'],b['expert_h2d_bytes'])

def main():
 manifest=json.loads((P/'execution_manifest.json').read_text())
 for path,h in manifest['source_sha256'].items():assert sha(path)==h,path
 for path,h in manifest['baseline_sha256'].items():assert sha(path)==h,path
 stages=sorted((json.loads(f.read_text()) for f in P.glob('R02_B*.json')),key=lambda d:d['batch'])
 assert any(d['batch']==8 for d in stages) and all(d['status']=='PASS' for d in stages)
 pairs=[];horizons=[];upper=[];occurrence_distributions={};r3points=[]
 for stage in stages:
  b=stage['batch'];horizon_acc={str(h):dict(occurrences=0,positive_future_occurrences=0,overlapping_future_marginal_sum=0,reuse_steps=0,reuse_routes=0,demand_steps=0) for h in (1,2,4,'remaining')}
  distances=Counter();bursts=Counter();demand_bursts=Counter()
  path=ROOT/f'reuse_occurrences_B{b}.jsonl.gz'
  assert sha(path)==next(a['sha256'] for a in stage['artifacts'] if a['path']==str(path))
  with gzip.open(path,'rt') as f:
   for line in f:
    row=json.loads(line);distances[str(row['next_remote_reuse_distance'])]+=1;bursts[str(row['remote_burst'])]+=1;demand_bursts[str(row['demand_burst'])]+=1
    for h,v in row['horizons'].items():
     acc=horizon_acc[h];acc['occurrences']+=1;acc['positive_future_occurrences']+=v['future_peer_bytes_saved']>0;acc['overlapping_future_marginal_sum']+=v['future_peer_bytes_saved']
     for k in ('reuse_steps','reuse_routes','demand_steps'):acc[k]+=v[k]
  for h,a in horizon_acc.items():
   n=a['occurrences'];horizons.append(dict(batch=b,horizon=h,remote_occurrences=n,positive_future_fraction=a['positive_future_occurrences']/n,mean_future_marginal_bytes=a['overlapping_future_marginal_sum']/n,mean_future_remote_steps=a['reuse_steps']/n,mean_future_remote_routes=a['reuse_routes']/n,mean_future_demand_steps=a['demand_steps']/n,next_remote_recurrence_weighted_byte_share=stage['marginal_byte_share_next_remote_reuse'].get(h)))
  occurrence_distributions[b]=dict(next_remote_step_distance_counts=distances,consecutive_remote_burst_counts=bursts,consecutive_positive_demand_burst_counts=demand_bursts)
  row=dict(batch=b,pair_count=stage['pair_count'],actual_F_peer_bytes=stage['baseline']['peer_activation_bytes'],individual_marginal_peer_bytes=stage['remote_marginal_bytes'],R2_gate=stage['R2_gate'])
  for metric,shares in stage['concentration'].items():
   row[metric+'_gini']=stage['gini'][metric]
   for fraction,share in shares.items():row[metric+'_top_'+fraction+'_share']=share
  for h,v in stage['marginal_byte_share_next_remote_reuse'].items():row['marginal_byte_share_recur_'+h]=v
  pairs.append(row);upper+=stage['persistence']
  f=P/f'R3_B{b}.json'
  if f.exists():
   d=json.loads(f.read_text());assert d['status']=='PASS' and len(d['points'])==16
   assert {(str(r['horizon']),r['threshold_peer_bytes']) for r in d['points']}=={(str(h),t) for h in (1,2,4,'remaining') for t in (0,65536,262144,1048576)}
   for point in d['points']:
    assert point['events']==384 and point['total_fetches']==sum(point[k] for k in ('first_copy_fetches','reload_fetches','replica_fetches'))
    assert point['expert_h2d_bytes']==point['total_fetches']*9437184
    assert point['peer_activation_bytes']==point['dispatch_bytes']+point['combine_bytes']
    life=point['lifetimes'];assert Path(life['path']).stat().st_size==life['bytes'] and sha(life['path'])==life['sha256']
    count=reused=censored=0;durations=[]
    with gzip.open(life['path'],'rt') as stream:
     for text in stream:
      copy=json.loads(text);count+=1;reused+=copy['reused'];censored+=copy['right_censored'];durations.append(copy['lifetime_layer_events'])
      assert copy['lifetime_layer_events']==copy['end_event']-copy['birth_event']>0
      assert bool(copy['reuse_events'])==copy['reused']
    assert count==point['replica_admissions']==point['replica_fetches'] and reused==point['reused_replicas'] and censored==point['right_censored_replicas']
    assert math.isclose(point['fraction_reused_before_eviction'],reused/count if count else 0)
    assert math.isclose(point['mean_replica_lifetime_layer_events'],statistics.mean(durations) if count else 0)
    ordered=sorted(durations)
    point['replica_lifetime_p50_layer_events']=statistics.median(ordered) if count else None
    if count:
     pos=(count-1)*.9;lo=math.floor(pos);hi=math.ceil(pos)
     point['replica_lifetime_p90_layer_events']=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
    else:point['replica_lifetime_p90_layer_events']=None
   r3points+=d['points']
 csvwrite('reuse_pair_summary.csv',pairs);write('reuse_pair_summary.json',dict(rows=pairs,per_pair_files=[dict(path=f'reuse_pairs_B{d["batch"]}.csv',sha256=sha(P/f'reuse_pairs_B{d["batch"]}.csv')) for d in stages],raw_accounting=[dict(batch=d['batch'],baseline=d['baseline'],artifacts=d['artifacts'],provenance=d['provenance']) for d in stages]))
 csvwrite('reuse_horizon_summary.csv',horizons);write('reuse_horizon_summary.json',dict(rows=horizons,occurrence_distributions=occurrence_distributions,future_definition='Strictly later same-layer decode steps; eight-step right truncation; current event excluded.'))
 csvwrite('persistence_upper_bound.csv',upper);write('persistence_upper_bound.json',dict(rows=upper,per_candidate_files=[a for d in stages for a in d['artifacts'] if 'persistence_candidates' in a['path']],interpretation='Independent candidate persistence bounds; sums are not a jointly attainable multi-replica saving or time model.'))
 if r3points:
  csvwrite('selective_replay_points.csv',[{k:v for k,v in r.items() if k!='lifetimes'} for r in r3points]);write('selective_replay_points.json',dict(status='PASS',points=r3points,policy='Future-aware frozen-F marginal scoring with fetch-normalized victim penalty, exact physical slots/LRU; diagnostic, not globally optimal.'))
 old=json.loads((OLD/'replica_pareto_screen.json').read_text())['points'];oldrows=[]
 for r in old:
  oldrows.append(dict(label={0:'F',.25:'K',.75:'C'}.get(r['rho'],f"rho={r['rho']}"),rho=r['rho'],**{k:r['decode'][k] for k in ('expert_h2d_bytes','peer_activation_bytes')}))
 F=next(r for r in oldrows if r['rho']==0);K=next(r for r in oldrows if r['rho']==.25);points8=[r for r in r3points if r['batch']==8]
 frontier=[]
 for r in oldrows:frontier.append(dict(kind='old',label=r['label'],horizon=None,threshold_peer_bytes=None,peer_activation_bytes=r['peer_activation_bytes'],expert_h2d_bytes=r['expert_h2d_bytes'],dominates_old_rhos=[],strong_option1=False,strong_option2=False))
 for r in points8:
  peer,h2d=r['peer_activation_bytes'],r['expert_h2d_bytes']
  frontier.append(dict(kind='selective',label=f"H{r['horizon']}/T{r['threshold_peer_bytes']//1024}KiB",horizon=r['horizon'],threshold_peer_bytes=r['threshold_peer_bytes'],peer_activation_bytes=peer,expert_h2d_bytes=h2d,
    dominates_old_rhos=[q['rho'] for q in oldrows if q['rho']>0 and dominates(r,q)],strong_option1=peer<=K['peer_activation_bytes'] and 4*h2d<=3*K['expert_h2d_bytes'],strong_option2=4*h2d<=3*F['expert_h2d_bytes']+K['expert_h2d_bytes'] and 2*(F['peer_activation_bytes']-peer)>=F['peer_activation_bytes']-K['peer_activation_bytes']))
 for r in frontier:r['nondominated']=not any(dominates(q,r) for q in frontier)
 primary=next(d for d in stages if d['batch']==8)
 if not primary['R2_gate']:decision='LOW_RANK_LOCAL_REUSE'
 elif not points8:decision='CAPACITY_REPLAY_PENDING'
 elif any(r['strong_option1'] or r['strong_option2'] for r in frontier if r['kind']=='selective'):decision='STRONG_HEADROOM'
 elif any(r['dominates_old_rhos'] for r in frontier):decision='MODEST_HEADROOM'
 else:decision='NO_HEADROOM'
 csvwrite('frontier_comparison.csv',frontier);write('frontier_comparison.json',dict(decision=decision,rows=frontier,primary_batch=8,secondary_reference_frontiers='Not available; none fabricated.'))
 if points8:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
  ax.plot([r['peer_activation_bytes']/2**20 for r in oldrows],[r['expert_h2d_bytes']/2**30 for r in oldrows],'o-',label='Committed rho frontier',color='#6b7280')
  for r in oldrows:ax.annotate(r['label'],(r['peer_activation_bytes']/2**20,r['expert_h2d_bytes']/2**30),xytext=(5,5),textcoords='offset points',fontsize=9)
  for h,c in zip((1,2,4,'remaining'),('#2563eb','#16a34a','#f97316','#9333ea')):
   cells=[r for r in points8 if r['horizon']==h];ax.scatter([r['peer_activation_bytes']/2**20 for r in cells],[r['expert_h2d_bytes']/2**30 for r in cells],label=f'H={h} (4 thresholds)',color=c,s=55,alpha=.8)
  ax.set(xlabel='Decode peer activation traffic (MiB)',ylabel='Decode expert H2D (GiB)',title=f'B8 rank-local selective replication: {decision}')
  ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(P/'frontier_B8.png',dpi=180);fig.savefig(P/'frontier_B8.svg');plt.close(fig)
  svg=P/'frontier_B8.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 lines=['# Rank-local reuse and selective replication headroom','',f'**{decision}** — CPU accounting only; no E2E speedup claim.','',
 'The B8 rho0 replay exactly reproduced F: 17,635 fetches, 166,424,739,840 expert H2D bytes, 334,970,880 peer bytes and 23,179 remote token-rank pairs. Destinations, traffic matrices, fetch classes and full physical cache/LRU state matched the independent reference at all 432 events.','',
 '## Rank-local reuse','', '| Local batch | Observed remote pairs | Top 10% marginal byte share | Byte share recurring within 4 steps | Gate |','|---:|---:|---:|---:|:---|']
 for d in stages:lines.append(f"| {d['batch']} | {d['pair_count']} | {d['concentration']['marginal_peer_bytes']['0.1']:.2%} | {d['marginal_byte_share_next_remote_reuse']['4']:.2%} | {'PASS' if d['R2_gate'] else 'FAIL'} |")
 lines+=['','The gate weights current **individual candidate marginal** bytes, not total wire traffic. Shared dispatch rows create interactions, so individual marginals are not additive joint savings. Recurrence is truncated at the eighth decode step; current-event savings are excluded from all future scores.','', '## Persistence-only upper bound','', '| Batch | Horizon | Positive candidates / all | Maximum future peer saving (KiB) | Max peer/fetch byte ratio | Byte break-even candidates |','|---:|:---|:---|---:|---:|---:|']
 for r in upper:lines.append(f"| {r['batch']} | {r['horizon']} | {r['positive_saving_candidates']} / {r['candidates']} | {r['max_future_peer_bytes']/1024:.1f} | {r['max_peer_per_fetch_byte_ratio']:.5f} | {r['break_even_candidates']} |")
 lines+=['','One hypothetical replica costs 9 MiB. These ratios are byte accounting, not equivalent time costs. Each bound assumes a free, persistent replica for one pair; sums of independent candidates are not a joint oracle policy.','', '## Capacity-aware replay','']
 if points8:
  lines+=['Exactly 16 predeclared horizon/threshold cells were run per available batch. All use original physical slots/LRU and protected active experts. Future scores are frozen F marginals; a unique victim needed within H adds one projected fetch, dividing the score by two. Strict score > threshold admits a copy; ties use expert ID then rank. Prefill is identical to F without optional copies. Old K/C have historical greedy prefills, so their decode-start states may differ.','', '| B8 cell | Peer MiB | H2D GiB | Replica admissions | Later local reuse | Victim reloads | Dominated old rho |','|:---|---:|---:|---:|---:|---:|:---|']
  for r in points8:
   f=next(x for x in frontier if x['kind']=='selective' and x['horizon']==r['horizon'] and x['threshold_peer_bytes']==r['threshold_peer_bytes'])
   lines.append(f"| {f['label']} | {r['peer_activation_bytes']/2**20:.3f} | {r['expert_h2d_bytes']/2**30:.3f} | {r['replica_admissions']} | {r['fraction_reused_before_eviction']:.2%} | {r['victim_induced_reload_count']} | {f['dominates_old_rhos']} |")
  zero=[r for r in points8 if r['threshold_peer_bytes']==0]
  lines += ['', f"At threshold zero, only {min(r['reused_replicas'] for r in zero)}–{max(r['reused_replicas'] for r in zero)} replicas per cell serve a later local demand out of {min(r['replica_admissions'] for r in zero):,}–{max(r['replica_admissions'] for r in zero):,} admissions. Mean copy lifetimes are {min(r['mean_replica_lifetime_layer_events'] for r in zero):.2f}–{max(r['mean_replica_lifetime_layer_events'] for r in zero):.2f} layer events, versus 48 layer events to the next same-layer decode opportunity."]
  lines+=['','![B8 frontier](frontier_B8.png)','', 'Replica reuse counts only a later local service before eviction; current-event service is excluded. End-of-trace survivors are right-censored. All lifecycle counts and actual coordinates, including secondary B16/B32, are in the selective replay files. No secondary historical frontier is invented.']
 else:lines+=['No selective replay has run. Follow the gate and stage boundary above.']
 if any(r['batch']!=8 for r in r3points):
  lines += ['', '## Secondary batch coordinates', '', 'No historical K/C points exist for these batches. The table shows each freshly reproduced F and the selective cell with the lowest peer bytes; this is not a secondary headroom classification.', '', '| Batch | F peer MiB | F H2D GiB | Lowest-peer cell | Selective peer MiB | Selective H2D GiB |', '|---:|---:|---:|:---|---:|---:|']
  for b in (16,32):
   available=[r for r in r3points if r['batch']==b]
   if not available:continue
   base=next(d['baseline'] for d in stages if d['batch']==b);best=min(available,key=lambda r:(r['peer_activation_bytes'],r['expert_h2d_bytes']))
   lines.append(f"| {b} | {base['peer_activation_bytes']/2**20:.3f} | {base['expert_h2d_bytes']/2**30:.3f} | H{best['horizon']}/T{best['threshold_peer_bytes']//1024}KiB | {best['peer_activation_bytes']/2**20:.3f} | {best['expert_h2d_bytes']/2**30:.3f} |")
 if decision=='NO_HEADROOM':lines+=['','Temporal demand recurs, but this bounded selective policy does not dominate any old nonzero-rho point once real slots, evictions and reloads are included. This does not prove that every possible selective policy is useless; it does not justify an online controller or GPU follow-up.']
 elif decision in ('MODEST_HEADROOM','STRONG_HEADROOM'):lines+=['','The predeclared CPU frontier criterion is met by the cells marked in frontier_comparison.json. This is future-aware diagnostic headroom, not an online policy or measured latency benefit. Owner review is required before any GPU follow-up.']
 elif decision=='LOW_RANK_LOCAL_REUSE':lines+=['','The reuse gate failed. Stop without capacity-aware replay or a new controller.']
 maxrss=max([d['peak_rss_mib'] for d in stages]+[json.loads(f.read_text())['peak_rss_mib'] for f in P.glob('R3_B*.json')])
 lines+=['','## Validation and stage boundary','',f'- Peak study process RSS: {maxrss:.2f} MiB, under a hard 4-GiB address-space limit; CUDA hidden and BLAS/OMP threads fixed at one.',
 '- Eleven targeted/cache regression tests passed before tracing. Raw receipt sizes/SHA256 and committed cross-rank provenance were verified. Original CPU Pareto artifacts remain hash-identical.',
 '- No model, Torch, CUDA or NCCL work was launched by this study. The owner’s eight resident-model GPU worker PIDs remained unchanged.',
 '- Full per-occurrence, per-candidate, physical-state and replica-lifetime records are compressed outside Git with hashes. Compact summaries and per-pair statistics are checked in.',
 '- Eight decode steps bound observed reuse and replica lifetimes. Frozen-F scoring is not a globally optimal capacity oracle. No new threshold, horizon, repeat, cache ratio or physical F/K/C run was added.',
 '- Stop research work for owner review. Keep the requested resident-model idle load running.', '',
 'See [pair summaries](reuse_pair_summary.json), [horizons](reuse_horizon_summary.json), [persistence bounds](persistence_upper_bound.json), [frontier comparison](frontier_comparison.json), [validation](validation.json), and [execution conventions](EXECUTION_PROTOCOL.md).','']
 (P/'RESULTS.md').write_text('\n'.join(lines))
 workerroot=Path('/home/hwlee/mgo-results/model_inference_load_20261003');current=json.loads((workerroot/'processes.json').read_text());assert current==manifest['gpu_worker_pids_before']
 assert all('model_inference_load.py' in Path(f"/proc/{r['pid']}/cmdline").read_bytes().decode() for r in current)
 live_receipts=[json.loads((workerroot/f"gpu{r['gpu']}.json").read_text()) for r in current]
 assert all(s['pid']==r['pid'] and s['batch']==1024 and time.time()-s['unix']<120 for r,s in zip(current,live_receipts))
 outputs=[f for f in P.iterdir() if f.suffix in ('.csv','.json','.png','.svg') and f.name!='validation.json']+[P/'RESULTS.md']
 validation=dict(status='PASS',decision=decision,batches=[d['batch'] for d in stages],independent_F_events=432*len(stages),selective_cells=len(r3points),selective_decode_events=384*len(r3points),unit_tests=11,cpu_only=True,cuda_visible_devices='',address_space_limit_bytes=4*1024**3,threads=1,peak_rss_mib=maxrss,gpu_workers_unchanged=True,gpu_worker_pids=current,gpu_worker_live_receipts=[{k:r[k] for k in ('gpu','pid','batch','iterations','unix')} for r in live_receipts],baseline_hashes_unchanged=True,source_hashes_unchanged=True,new_research_gpu_runs=0,summary_source_sha256=sha(Path(__file__)),output_sha256={f.name:sha(f) for f in outputs})
 write('validation.json',validation)
 print(json.dumps({k:validation[k] for k in ('status','decision','batches','selective_cells','peak_rss_mib','gpu_workers_unchanged')},indent=2))
if __name__=='__main__':main()
