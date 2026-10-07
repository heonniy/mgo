"""Separate clean TPOT from diagnostic MoE time, volume and rank skew."""
import json,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]/'experiments/br_prefetch_c60_20261007'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
def analyze_batch(batch):
 labels=json.loads((P/'RUN_LABELS.json').read_text()) if (P/'RUN_LABELS.json').exists() else {}
 root=ROOT/labels.get(str(batch),f'br_prefetch_C60_B{batch}_H256_v1');out=P/f'B{batch}';out.mkdir(exist_ok=True);arms={};tokens={}
 for arm in ('off','on'):
  d=root/arm;primary=json.loads((d/'primary.json').read_text());assert primary['status']=='PASS'
  moe=np.zeros((4,256,48));expert=np.zeros_like(moe);expert_cpu=np.zeros_like(moe);rows=np.zeros_like(moe);groups=np.zeros_like(moe);miss=np.zeros_like(moe)
  arrivals={name:np.full_like(moe,np.nan) for name in ('return_token_a2a','forward_token_a2a_submit','moe.expert_compute')};rank_summaries=[];tokens[arm]=[]
  dest=out/arm;dest.mkdir(exist_ok=True)
  for name in ('primary.json','diagnostic.json'):shutil.copy2(d/name,dest/name)
  for rank in range(4):
   for name in (f'primary_rank{rank}.json',f'source_rank{rank}.json'):shutil.copy2(d/name,dest/name)
   pr=json.loads((d/f'primary_rank{rank}.json').read_text());tokens[arm]+=pr['tokens']
   x=json.loads((d/f'diagnostic_rank{rank}.json').read_text());assert x['token_cache_byte_parity'];totals=defaultdict(float);cpu=defaultdict(float)
   for e in x['moe_events']:
    if 1<=e['step']<=256:moe[rank,e['step']-1,e['layer']]=e['stream_seconds']
   for e in x['decode_cache_events']:
    s,l=e['step']-1,e['layer'];rows[rank,s,l]=e['token_expert_uses'];groups[rank,s,l]=e['expert_uses'];miss[rank,s,l]=e['demand_misses']
   for e in x['segments']:
    i=e['event_index']-48
    if not 0<=i<256*48:continue
    s,l=divmod(i,48);phase=e['phase'];totals[phase]+=e['stream_seconds'];cpu[phase]+=e['cpu_ns']/1e9
    if phase=='moe.expert_compute':expert[rank,s,l]+=e['stream_seconds'];expert_cpu[rank,s,l]+=e['cpu_ns']/1e9
    if phase in arrivals and np.isnan(arrivals[phase][rank,s,l]):arrivals[phase][rank,s,l]=e['host_timestamp_ns']/1e9
   byte_size=x['h2d_copies'][0]['bytes'];prefill_bytes=x['output']['prefill_bytes']['h2d']
   assert prefill_bytes%byte_size==0 and all(e['bytes']==byte_size for e in x['h2d_copies'])
   # Trace append and byte counter update share the scheduler lock. Prefill
   # drains required copies, with no speculative submissions, before boundary.
   decode_copy=x['h2d_copies'][prefill_bytes//byte_size:]
   assert sum(e['bytes'] for e in decode_copy)==pr['decode_bytes']['h2d']
   copy=np.array([e['service_seconds'] for e in x['h2d_copies']]);rank_summaries.append(dict(rank=rank,gpu=[0,1,4,5][rank],moe_seconds_per_token=float(moe[rank].sum()/256),phase_seconds_per_token={k:v/256 for k,v in totals.items()},cpu_seconds_per_token={k:v/256 for k,v in cpu.items()},expert_uses_per_token=float(groups[rank].sum()/256),token_expert_rows_per_token=float(rows[rank].sum()/256),demand_misses_per_token=float(miss[rank].sum()/256),h2d_service_seconds=float(copy.sum()),h2d_copy_ms_p50_p90_p99=[float(v) for v in np.percentile(copy*1000,[50,90,99])],primary_ready_metrics=pr['ready_metrics'],primary_decode_bytes=pr['decode_bytes'],decode_h2d_service_seconds=sum(e['service_seconds'] for e in decode_copy),decode_h2d_copy_ms_p50_p90_p99=np.percentile([e['service_seconds']*1000 for e in decode_copy],[50,90,99]).tolist()))
  assert np.all(rows.sum(axis=0)==batch*4*8) and np.all(moe>0)
  assert all(np.isfinite(a).all() for a in arrivals.values())
  skew={k:dict(mean_ms=float(np.ptp(a,axis=0).mean()*1000),p50_p90_p99_ms=[float(v) for v in np.percentile(np.ptp(a,axis=0)*1000,[50,90,99])]) for k,a in arrivals.items()}
  imbalance=dict(token_rows_max_over_mean_p50_p90=[float(v) for v in np.percentile(rows.max(0)/rows.mean(0),[50,90])],expert_count_max_over_mean_p50_p90=[float(v) for v in np.percentile(groups.max(0)/groups.mean(0),[50,90])],expert_span_max_minus_min_ms_p50_p90_p99=[float(v) for v in np.percentile(np.ptp(expert,axis=0)*1000,[50,90,99])],slowest_expert_rank_counts=np.bincount(expert.argmax(0).ravel(),minlength=4).tolist(),heaviest_token_rank_counts=np.bincount(rows.argmax(0).ravel(),minlength=4).tolist(),slowest_equals_heaviest_token_fraction=float(np.mean(expert.argmax(0)==rows.argmax(0))),slowest_equals_most_experts_fraction=float(np.mean(expert.argmax(0)==groups.argmax(0))))
  slow=expert.argmax(0)[None,:,:]
  imbalance['slowest_in_max_expert_count_ties_fraction']=float(np.mean(np.take_along_axis(groups,slow,axis=0)[0]==groups.max(0)))
  imbalance['slowest_in_max_token_rows_ties_fraction']=float(np.mean(np.take_along_axis(rows,slow,axis=0)[0]==rows.max(0)))
  centered=lambda a:(a-a.mean(0,keepdims=True)).ravel()
  imbalance['within_event_rank_deviation_correlations']={name:float(np.corrcoef(centered(expert_cpu),centered(a))[0,1]) for name,a in [('expert_cpu_vs_expert_count',groups),('expert_cpu_vs_token_rows',rows),('expert_cpu_vs_demand_misses',miss)]}
  controller=json.loads((d/'primary_rank0.json').read_text())['validation']['controller']
  arms[arm]=dict(primary=primary,controller=controller,h2d_service_scope='Unprefixed service fields include prefill; decode-prefixed fields exclude initial copies using the recorded prefill byte boundary. Copy service is overlapping, not additive to TPOT',diagnostic_moe_tpot=float(moe.sum(axis=2).max(axis=0).mean()),ranks=rank_summaries,host_collective_entry_skew=skew,imbalance=imbalance)
 for rank in range(4):
  source=[json.loads((root/arm/f'source_rank{rank}.json').read_text()) for arm in ('off','on')]
  assert source[0]['options']==source[1]['options'] and source[0]['capacities']==source[1]['capacities']==[920,920,919,919]
  assert source[0]['physical_slots']==source[1]['physical_slots']==[922,922,921,921]
  assert not source[0]['prefetch_enabled'] and source[1]['prefetch_enabled']
 off,on=arms['off'],arms['on'];agreement=float(np.mean(np.asarray(tokens['off'])==np.asarray(tokens['on'])))
 summary=dict(status='PASS',identical_capacity_overlap_verified=True,batch=batch,arms=arms,token_agreement_off_on=agreement,prefetch_tpot_gain_percent=(1-on['primary']['TPOT']/off['primary']['TPOT'])*100,peer_volume_reduction_percent=(1-on['primary']['decode_peer_bytes']/off['primary']['decode_peer_bytes'])*100,scope='one clean primary per arm; diagnostic MoE is a separate instrumented measurement; native BF16 routes may differ between arms')
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

def report_all():
 results=[json.loads(p.read_text()) for p in sorted(P.glob('B*/SUMMARY.json'))]
 lines=['# BR C60 prefetch physical results','','R4, input256,256 decode steps; identical MAIN3678 +8 reserved slots, overlap, ready-first and T2 in OFF/ON. OFF leaves prefetch slots unused. One clean primary and one separate full diagnostic per arm. No attention included in diagnostic MoE metric.','','|Batch|Prefetch|Clean TPOT s|Diagnostic MoE s/token|Decode peer GiB|Decode H2D GiB|','|---|---|---:|---:|---:|---:|']
 for x in sorted(results,key=lambda x:x['batch']):
  for name in ('off','on'):
   a=x['arms'][name];p=a['primary'];lines.append(f"|{x['batch']}|{name}|{p['TPOT']:.6f}|{a['diagnostic_moe_tpot']:.6f}|{p['decode_peer_bytes']/2**30:.3f}|{p['decode_h2d_bytes']/2**30:.3f}|")
 lines+=['','MoE metric = mean over steps of max-rank sum of48 MLP event durations. Includes router, host gaps, collectives and peer waiting; not pure GPU kernel time. Diagnostic overhead means it must not be subtracted from clean TPOT. Peer payload counts dispatch+return remote sends once, excluding metadata and NCCL protocol overhead.','', '|Batch|ON TPOT reduction %|ON peer volume reduction %|OFF/ON token agreement %|','|---|---:|---:|---:|']
 for x in sorted(results,key=lambda x:x['batch']):lines.append(f"|{x['batch']}|{x['prefetch_tpot_gain_percent']:.2f}|{x['peer_volume_reduction_percent']:.2f}|{100*x['token_agreement_off_on']:.2f}|")
 lines+=['','Per-rank phases, CPU execution time, expert/token workload and collective CPU-entry skew are in each batch SUMMARY.json. Collective entry timestamps share the host monotonic clock; they are not NCCL GPU start timestamps. H2D service overlaps compute and cannot be added to primary TPOT. Prefetch diagnostics match their arm primary tokens/cache/bytes. One primary is descriptive, not a stability confirmation. Only BR is measured: another admission policy cannot be declared a physical winner from this packet.']
 (P/'RESULTS.md').write_text('\n'.join(lines)+'\n');return results

def diagnostic_tables(results):
 lines=['# Rank-level evidence (BR C60)', '',
 'R4 ranks 0/1/2/3 map to GPUs 0/1/4/5. Batch is per rank: global batch is four times the label. Every time below is from the separate diagnostic, except explicitly marked primary counters. Decode totals are divided by 256, and include all 48 layers per step. Expert span includes CPU submission/gather work and GPU completion; it is not isolated GEMM time. Return span includes peer waiting. Phase columns are selected exclusive portions, not a complete partition.', '']
 for x in sorted(results,key=lambda x:x['batch']):
  lines += [f"## B{x['batch']}", '', '|Prefetch|Rank / GPU|Expert groups/step|Token-expert rows/step|Demand misses/step|Expert span ms/step|Expert thread CPU ms/step|Return collective ms/step|Metadata ms/step|', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
  for name,a in x['arms'].items():
   for r in a['ranks']:
    ph=r['phase_seconds_per_token'];cpu=r['cpu_seconds_per_token']
    lines.append(f"|{name}|{r['rank']} / {r['gpu']}|{r['expert_uses_per_token']:.1f}|{r['token_expert_rows_per_token']:.1f}|{r['demand_misses_per_token']:.1f}|{1000*ph['moe.expert_compute']:.2f}|{1000*cpu['moe.expert_compute']:.2f}|{1000*ph['return_token_a2a']:.2f}|{1000*ph['moe.metadata']:.2f}|")
  lines += ['', '|Prefetch|Dispatch entry skew mean ms/layer|Expert entry skew mean ms/layer|Return entry skew mean ms/layer|Expert span spread p50 / p90 ms/layer|Slowest = most experts % (ties allowed)|Slowest = most token rows % (ties allowed)|', '|---|---:|---:|---:|---:|---:|---:|']
  for name,a in x['arms'].items():
   sk=a['host_collective_entry_skew'];im=a['imbalance'];spread=im['expert_span_max_minus_min_ms_p50_p90_p99']
   lines.append(f"|{name}|{sk['forward_token_a2a_submit']['mean_ms']:.3f}|{sk['moe.expert_compute']['mean_ms']:.3f}|{sk['return_token_a2a']['mean_ms']:.3f}|{spread[0]:.3f} / {spread[1]:.3f}|{100*im.get('slowest_in_max_expert_count_ties_fraction',im['slowest_equals_most_experts_fraction']):.1f}|{100*im.get('slowest_in_max_token_rows_ties_fraction',im['slowest_equals_heaviest_token_fraction']):.1f}|")
  lines += ['', 'Entry skew is max minus min host API-entry timestamp across four ranks for the same layer/step, not GPU network start time. Matching the heaviest rank is observational; ties are allowed and become common at large batches.', '', '|Prefetch|Issued|Useful|Wasted|Useful / issued %|Mandatory fetches|Promotion-victim reloads|Primary ready-first waits (sum ranks)|', '|---|---:|---:|---:|---:|---:|---:|---:|']
  for name,a in x['arms'].items():
   c=a.get('controller')
   if c is None:continue
   useful=f"{100*c['useful']/c['issued']:.2f}" if c['issued'] else 'N/A'
   lines.append(f"|{name}|{c['issued']}|{c['useful']}|{c['wasted']}|{useful}|{c['mandatory']}|{c['promotion_victim_reloads']}|{sum(r['primary_ready_metrics']['waits'] for r in a['ranks'])}|")
  lines += ['', 'Controller counters are global primary counters including prefill; the speculative mechanism is used during decode. Useful is consumed prefetch, not a net saved-copy count. Promotion-victim reloads are events, not proof that all are additional misses caused by prefetch. Ready-first waits count calls taking an explicit not-ready path; zero does not mean H2D takes zero time.', '', '|Prefetch|Rank|Primary decode peer GiB|Primary decode H2D GiB|Decode H2D service p50 / p90 / p99 ms per copy|', '|---|---:|---:|---:|---:|']
  for name,a in x['arms'].items():
   for r in a['ranks']:
    b=r.get('primary_decode_bytes')
    if b is None:continue
    q=r['decode_h2d_copy_ms_p50_p90_p99']
    lines.append(f"|{name}|{r['rank']}|{(b['forward']+b['returned'])/2**30:.3f}|{b['h2d']/2**30:.3f}|{q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}|")
  lines += ['', 'Decode H2D percentiles exclude the initial copies using the recorded prefill byte boundary; their byte sum is checked against primary decode H2D bytes. They measure copy service, not queue delay or exposed critical-path wait.', '']
 lines += ['## Interpretation boundaries', '', 'OFF and ON use identical MAIN and reserved capacities, streaming, ready-first and T2 synchronization. Only speculative prefetch submission is disabled in OFF. Native generated tokens differ across arms, so subsequent routing can differ. Detailed diagnostics reproduce their own primary tokens/cache/bytes. No frozen-route counterfactual or repeated timing confirmation is claimed.', '', 'Unprefixed H2D service fields in SUMMARY.json include prefill; decode-prefixed fields exclude it. Copy-stream times overlap other work. Demand-H2D submission time is not DMA duration. Do not add service time to TPOT or subtract diagnostic MoE time from clean TPOT.', '', 'Placement recommendations must distinguish per-expert executor overhead, token-row work, exposed fetch dependency, and peer waiting. BR is the only measured placement; these observations cannot establish a measured LA/CA/FCA/Near winner.']
 (P/'RANK_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--batch',type=int,choices=(8,16,64));a=p.parse_args()
 if a.batch:analyze_batch(a.batch)
 diagnostic_tables(report_all())
