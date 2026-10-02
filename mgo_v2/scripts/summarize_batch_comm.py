"""Apply frozen batch geometry, whole-trace timing, and descriptive diagnostics."""
import json,math,statistics
from pathlib import Path
from batch_comm_common import PACKET,ROOT,BATCHES,ORDER,write,sha
from batch_comm_analysis import csvwrite,predictors,summary
from summarize_trace_comm_replay import percentile

def median(xs):return statistics.median(xs)
def corr(x,y):
 mx,my=statistics.mean(x),statistics.mean(y);vx=sum((v-mx)**2 for v in x);vy=sum((v-my)**2 for v in y)
 return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(vx*vy) if vx and vy else None

def main():
 progress=json.loads((PACKET/'batch_comm_progress.json').read_text());assert progress['status']=='STAGE_PASS' and 'timing' in progress['completed']
 assert all(sha(p)==h for p,h in progress['source_sha256'].items())
 assert all(sha(PACKET/n)==h for n,h in progress['cpu_result_sha256'].items())
 source={b:json.loads((PACKET/'trace_comm_counts.json' if b==8 else ROOT/f'counts_B{b}.json').read_text()) for b in BATCHES}
 cells=[];all_events={};raw_receipts=[];memory=[]
 for folder in sorted(ROOT.iterdir()):
  if folder.is_dir():
   state=json.loads((folder/'status.json').read_text());assert state['status']=='PASS';memory+=state['memory']
   for path in sorted(folder.iterdir()):
    if path.is_file():raw_receipts.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
 for b in BATCHES:
  expected_hash=sha(PACKET/'trace_comm_counts.json' if b==8 else ROOT/f'counts_B{b}.json')
  if b!=8:assert expected_hash==json.loads((PACKET/f'batch_comm_capture_B{b}.json').read_text())['counts_sha256']
  for p,mode in ORDER:
   rs=[json.loads((ROOT/f'B{b}_p{p}_{mode}'/f'rank{r}.json').read_text()) for r in range(4)]
   env={'NCCL_CUMEM_ENABLE':'0'}
   if mode=='R3':env.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
   assert all(r['status']=='PASS' and r['rank']==i and r['counts_sha256']==expected_hash and r['transport_env']==env and r['payload_valid'] and r['ipc_rebase'] and r['no_model'] and r['events']==384 and r['warmup_full_traces']==1 and r['timed_full_traces']==3 and not r['nccl_info_in_timing'] for i,r in enumerate(rs))
   for phase in ('dispatch','combine'):assert sum(r['peer_bytes_per_trace'][phase] for r in rs)==source[b]['peer_bytes_per_trace'][phase]
   metrics={k:[] for k in ('whole_cuda_ms','whole_wall_ms','sum_event_max_rank_ms','max_rank_sum_event_ms')};dist={ph:[] for ph in ('dispatch','combine','pair')}
   for rep in range(3):
    repeats=[r['repeats'][rep] for r in rs];assert all(r['payload_valid'] and len(r['events_ms'])==384 for r in repeats)
    maxima=[{ph:max(r['events_ms'][i][ph] for r in repeats) for ph in dist} for i in range(384)]
    metrics['whole_cuda_ms'].append(max(r['full_trace_cuda_interval_ms'] for r in repeats));metrics['whole_wall_ms'].append(max(r['wall_ms'] for r in repeats))
    metrics['sum_event_max_rank_ms'].append(sum(e['pair'] for e in maxima));metrics['max_rank_sum_event_ms'].append(max(r['cumulative_pair_ms'] for r in repeats))
    for i,e in enumerate(maxima):
     for ph in dist:dist[ph].append(e[ph])
     all_events.setdefault((b,mode,i),[]).append(e['pair'])
   row=dict(local_batch=b,global_batch=4*b,pass_index=p,mode=mode)
   row.update({k:median(v) for k,v in metrics.items()})
   for ph,values in dist.items():
    for pct in (50,90,99):row[f'{ph}_p{pct}_ms']=percentile(values,pct)
   cells.append(dict(**row,raw_totals_ms=metrics))
 csvwrite('batch_comm_trace_timing.csv',[{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in cells])
 ratios=[];by={(r['local_batch'],r['pass_index'],r['mode']):r for r in cells}
 for b in BATCHES:
  row=dict(local_batch=b,global_batch=4*b)
  for k in ('whole_cuda_ms','whole_wall_ms','sum_event_max_rank_ms','max_rank_sum_event_ms'):
   values=[by[b,p,'R3'][k]/by[b,p,'T0'][k] for p in (0,1)];row[k+'_pass_ratios']=values;row[k+'_median_ratio']=median(values)
  ratios.append(row)
 write(PACKET/'batch_comm_trace_timing.json',dict(status='PASS',primary='max rank whole-trace CUDA and wall, median three repeats; median of within-pass ratios',cells=cells,ratios=ratios,raw_receipts=raw_receipts))
 # Small payload calibration: paired rank maxima per iteration; no averaging ranks.
 latency=[];pool={}
 for p,mode in ORDER:
  rs=[json.loads((ROOT/f'latency_p{p}_{mode}'/f'rank{r}.json').read_text()) for r in range(4)]
  for r in rs:assert r['status']=='PASS' and len(r['results'])==5 and not r['smoke']
  for i,size in enumerate((16384,32768,65536,131072,262144)):
   assert all(r['results'][i]['payload_valid'] and r['results'][i]['per_peer_payload_bytes']==size and len(r['results'][i]['cuda_interval_ms'])==30 for r in rs)
   values=[max(r['results'][i]['cuda_interval_ms'][j] for r in rs) for j in range(30)]
   latency.append(dict(pass_index=p,mode=mode,per_peer_bytes=size,median_ms=median(values),p90_ms=percentile(values,90),raw_max_rank_ms=values))
   pool.setdefault((mode,size),[]).extend(values)
 fits=[]
 for mode in ('T0','R3'):
  x=[16384,32768,65536,131072,262144];y=[median(pool[mode,s]) for s in x];mx,my=statistics.mean(x),statistics.mean(y)
  beta=sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x);alpha=my-beta*mx
  residual=sum((b-alpha-beta*a)**2 for a,b in zip(x,y));var=sum((b-my)**2 for b in y)
  fits.append(dict(mode=mode,alpha_ms=alpha,beta_ms_per_byte=beta,r_squared=1-residual/var if var else None,points=[dict(bytes=a,median_ms=b) for a,b in zip(x,y)]))
 csvwrite('small_payload_latency.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in latency])
 write(PACKET/'small_payload_latency.json',dict(status='PASS',cells=latency,fits=fits,interpretation='Descriptive overhead/byte-sensitivity proxy, not physical link bandwidth.'))
 # Buckets per batch and combined; unique event denominators, pooled repeated intervals.
 names=['<=32 KiB','32-64 KiB','64-128 KiB','128-256 KiB','>256 KiB']
 def bucket(v):return next((i for i,limit in enumerate((32768,65536,131072,262144)) if v<=limit),4)
 preds={(b,i):predictors(e) for b in BATCHES for i,e in enumerate(source[b]['events'])}
 buckets=[];correlations=[]
 for batch in (*BATCHES,'all'):
  keys=[key for key in preds if batch=='all' or key[0]==batch];totalbytes=sum(preds[k]['total_peer_bytes'] for k in keys)
  for number,label in enumerate(names):
   chosen=[k for k in keys if bucket(preds[k]['max_rank_remote_bytes'])==number]
   if not chosen:continue
   values={mode:[v for b,i in chosen for v in all_events[b,mode,i]] for mode in ('T0','R3')}
   row=dict(local_batch=batch,bucket=label,unique_events=len(chosen),event_fraction=len(chosen)/len(keys),peer_bytes=sum(preds[k]['total_peer_bytes'] for k in chosen),peer_byte_fraction=sum(preds[k]['total_peer_bytes'] for k in chosen)/totalbytes)
   for mode,vs in values.items():row.update({mode+'_pair_median_ms':median(vs),mode+'_pair_p90_ms':percentile(vs,90)})
   row['R3_over_T0_median_ratio']=row['R3_pair_median_ms']/row['T0_pair_median_ms'];buckets.append(row)
  for mode in ('T0','R3'):
   ys=[median(all_events[b,mode,i]) for b,i in keys]
   correlations.append(dict(local_batch=batch,mode=mode,events=len(keys),**{key:corr([preds[k][key] for k in keys],ys) for key in ('max_rank_remote_bytes','total_peer_bytes','active_remote_ordered_pairs','max_rank_fanout')}))
 csvwrite('batch_comm_event_buckets.csv',buckets);write(PACKET/'batch_comm_event_diagnostics.json',dict(buckets=buckets,pearson_correlations=correlations))
 geom=json.loads((PACKET/'batch_comm_geometry.json').read_text())['rows'];g={r['local_batch']:r for r in geom if r['phase']=='union'};rat={r['local_batch']:r for r in ratios}
 interpretations=[]
 for end in (16,32):
  msg=g[end]['p50_message_bytes']/g[4]['p50_message_bytes'];mx=g[end]['max_rank_remote_bytes_p50']/g[4]['max_rank_remote_bytes_p50'];count=g[end]['nonzero_remote_messages']/g[4]['nonzero_remote_messages'];delta=rat[end]['whole_cuda_ms_median_ratio']-rat[4]['whole_cuda_ms_median_ratio']
  grows=max(msg,mx)>=1.10
  if grows and delta>=.10:decision='BATCH_SENSITIVE'
  elif grows and all(rat[b]['whole_cuda_ms_median_ratio']<=1.10 for b in BATCHES if b<=end):decision='LATENCY_DOMINATED'
  elif not grows and count>=1.10:decision='MORE_MESSAGES_NOT_BIGGER_MESSAGES'
  else:decision='MIXED'
  interpretations.append(dict(comparison=f'B4_to_B{end}',decision=decision,message_median_growth_ratio=msg,max_rank_bytes_median_growth_ratio=mx,message_count_growth_ratio=count,whole_cuda_ratio_increase_pp=delta*100,whole_wall_ratio_increase_pp=100*(rat[end]['whole_wall_ms_median_ratio']-rat[4]['whole_wall_ms_median_ratio'])))
 mem=dict(peak_process_tree_rss_gib=max(s.get('group_rss_bytes',0) for s in memory)/2**30,min_host_available_gib=min(s['host_available_bytes'] for s in memory)/2**30,min_target_gpu_free_mib=min(g['free_mib'] for s in memory for g in s['gpu'].values()))
 lines=['# Batch communication sensitivity — B4 through B32','',f"**B4→B32: {interpretations[-1]['decision']}**; original B4→B16 endpoint: **{interpretations[0]['decision']}**.",'',
 'Local batches 4/8/16/32 correspond to global 16/32/64/128 on R4. Cache30, exact-only P0 seed42/LRU, no replicas. B8 source reused; exactly three new diagnostic model captures. All captures/counts and communication payloads passed. Model capture times are not performance evidence.','',
 '## Message geometry','', '| Local batch | Phase | Peer MiB | Nonzero messages | Message p50/p90/p99 KiB | Self fraction |','|---:|:---|---:|---:|:---|---:|']
 for r in geom:lines.append(f"| {r['local_batch']} | {r['phase']} | {r['peer_bytes']/2**20:.3f} | {r['nonzero_remote_messages']} | {r['p50_message_bytes']/1024:.1f} / {r['p90_message_bytes']/1024:.1f} / {r['p99_message_bytes']/1024:.1f} | {r['self_byte_fraction']:.3f} |")
 lines+=['','## Whole-trace primary timings','', '| Local batch | CUDA R3/T0 pass 0 / pass 1 | Median CUDA ratio | Median wall ratio |','|---:|:---|---:|---:|']
 for r in ratios:lines.append(f"| {r['local_batch']} | {' / '.join(f'{v:.4f}' for v in r['whole_cuda_ms_pass_ratios'])} | {r['whole_cuda_ms_median_ratio']:.4f} | {r['whole_wall_ms_median_ratio']:.4f} |")
 lines+=['','Per-repeat whole CUDA/wall times, both secondary aggregation definitions and dispatch/combine/pair p50/p90/p99 are retained in the timing CSV/JSON. Primary is max rank whole-trace interval per repeat, median of three; reported ratio summarizes two counter-ordered passes. Event sum-max is diagnostic only.','', '## Descriptive interpretation','']
 for r in interpretations:lines.append(f"- {r['comparison']}: {r['decision']}; union message median {r['message_median_growth_ratio']:.3f}x, median max-rank remote bytes {r['max_rank_bytes_median_growth_ratio']:.3f}x, nonzero message count {r['message_count_growth_ratio']:.3f}x. CUDA ratio changes {r['whole_cuda_ratio_increase_pp']:+.2f} percentage points; wall ratio {r['whole_wall_ratio_increase_pp']:+.2f} points.")
 lines+=['','These are bounded descriptive measurements, not confidence-tested causal evidence. Event intervals include launch/stream scheduling gaps. Whole-trace throughput and event-level latency can differ; the older B8 IPC/SHM result is historical context, not pooled into this run.','', '## Calibration and event diagnostics','']
 for fit in fits:lines.append(f"- {fit['mode']}: alpha={fit['alpha_ms']:.6f} ms, beta={fit['beta_ms_per_byte']:.3e} ms/byte, R²={fit['r_squared']:.3f}. Five per-peer sizes only; no physical bandwidth claim.")
 lines+=['','[Event buckets](batch_comm_event_buckets.csv) report pair median/p90, R3/T0, unique event fractions and byte fractions for each batch and pooled. [Diagnostics](batch_comm_event_diagnostics.json) include correlations with bytes and fan-out; constant predictors are null. Predictors are transport-independent and fixed before timing.','', '## Validation and limits','',
 f"- Peak process-tree RSS {mem['peak_process_tree_rss_gib']:.2f} GiB; minimum host available {mem['min_host_available_gib']:.2f} GiB; minimum target GPU free {mem['min_target_gpu_free_mib']:,} MiB. No memory guard stop.",
 '- All CPU Pareto results and frozen schedule metadata retain their SHA256 hashes. No CPU screen, F/K/C run, additional decode, NCCL tuning, or NVLink-mode operation.',
 '- Fresh transport smokes accepted T0 P2P/IPC and R3 SHM only. Both use NCCL_CUMEM_ENABLE=0. INFO is absent during timing.',
 '- Eight resident-model inference workers are paused throughout experiment stages and restarted on exit. Their safeguards yield to other GPU jobs and memory/temperature pressure. Idle inference is separate from the three research captures.',
 '- Raw capture/timing receipts are outside Git with hashes; compact geometry, prompt provenance, diagnostics and results are checked in. No automatic F/K model timing; stop for owner review.', '',
 'See [geometry](batch_comm_geometry.json), [trace timing](batch_comm_trace_timing.json), [calibration](small_payload_latency.json), [validation](batch_comm_validation.json), and [frozen conventions](BATCH_COMM_EXECUTION.md).','']
 (PACKET/'BATCH_COMM_SENSITIVITY_RESULTS.md').write_text('\n'.join(lines))
 outputs=['batch_comm_geometry.csv','batch_comm_geometry.json','batch_comm_trace_timing.csv','batch_comm_trace_timing.json','batch_comm_event_buckets.csv','batch_comm_event_diagnostics.json','small_payload_latency.csv','small_payload_latency.json','BATCH_COMM_SENSITIVITY_RESULTS.md']
 write(PACKET/'batch_comm_validation.json',dict(status='PASS',interpretations=interpretations,new_model_captures=3,B8_reused=True,decode_events_per_trace=384,
  trace_cells=16,timed_traces=48,warmup_traces=16,validated_global_trace_events=64*384,latency_cells=20,transport_smokes=2,
  cpu_results_unchanged=True,cpu_result_sha256=progress['cpu_result_sha256'],sources_unchanged=True,source_sha256=progress['source_sha256'],
  analysis_source_sha256={str(Path(__file__)):sha(Path(__file__)),str(PACKET.parents[1]/'scripts/batch_comm_analysis.py'):sha(PACKET.parents[1]/'scripts/batch_comm_analysis.py')},memory=mem,
  output_sha256={n:sha(PACKET/n) for n in outputs}))
 print(json.dumps(dict(interpretations=interpretations,ratios=ratios,memory=mem),indent=2))
if __name__=='__main__':main()
