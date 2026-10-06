"""Validate full primary receipts and summarize all attempts without cherry-picking."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
SYSTEMS=['Ours','MoE-Infinity-repaired','DeepSpeed-ZeRO-Inference','llama.cpp-layer']
def read(p):return json.loads(p.read_text())
def summarize(values):
 mean=statistics.mean(values);median=statistics.median(values)
 return dict(samples=values,median=median,min=min(values),max=max(values),cv_percent=statistics.stdev(values)/mean*100,spread_percent=(max(values)-min(values))/mean*100)
def check_clock(start,stamps):
 assert len(stamps)==64 and start<stamps[0]
 assert all(a<b for a,b in zip(stamps,stamps[1:]))
def check_manifests(spec):
 groups=[]
 for phase in ['target','warmup']:
  source=spec[phase];data=Path(source['path']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==source['sha256'],'frozen manifest changed'
  rows=json.loads(data)['requests'];assert len(rows)==spec['global_requests']
  assert len(set(r['request_id'] for r in rows))==len(rows)
  assert all(len(r['input_ids'])==spec['input_tokens'] for r in rows)
  groups.append(set(r['conversation_id'] for r in rows))
 assert not groups[0].intersection(groups[1]),'warmup conversations overlap target'
def audit(job,spec):
 result=read(job/'result.json');assert result['status']=='PASS' and not result.get('smoke',False)
 system=result['system'];expected=[x['request_id'] for x in read(Path(spec['target']['path']))['requests']]
 rows=[];host=[];pinned=[]
 for repeat in [1,2,3]:
  row=read(job/f'repeat{repeat}.json');assert row['status']=='PASS' and not row.get('smoke',False)
  assert row['output_tokens']==64 and row['global_requests']==spec['global_requests']
  assert all(math.isfinite(row[k]) and row[k]>0 for k in ['TTFT','TPOT','E2E','throughput'])
  assert abs(row['E2E']-row['TTFT']-63*row['TPOT'])<1e-6
  assert abs(row['throughput']-spec['global_requests']*64/row['E2E'])<1e-6
  if system in ['Ours','DeepSpeed-ZeRO-Inference']:
   ranks=[read(job/f'repeat{repeat}_rank{r}.json') for r in range(4)]
   actual=[i for rank in ranks for i in rank['request_ids']]
   assert len(set(rank['release_ns'] for rank in ranks))==1
   for rank in ranks:
    assert len(rank['tokens'])==spec['local_batch'] and all(len(t)==64 for t in rank['tokens'])
    assert rank['finite_logits']
    check_clock(rank['release_ns'],rank['token_ready_ns'])
    assert rank['first_ns']==rank['token_ready_ns'][0] and rank['end_ns']==rank['token_ready_ns'][-1]
    if system=='Ours':
     assert rank['policy']=='LA_CA_NEAR'
     assert rank['no_compile'] and rank['expert_cache_start']=='empty'
     assert rank['validation']['status']=='PASS' and rank['validation']['physical_slots']==1843
    else:
     assert rank['cache_start']=='all parameters NOT_AVAILABLE'
     assert rank['kv_gpu_resident'] and 0<rank['all_parameter_peak_bytes']<=rank['parameter_budget_bytes']
   start=ranks[0]['release_ns'];first=max(r['first_ns'] for r in ranks);end=max(r['end_ns'] for r in ranks)
   host.append(sum(r['host_rss_bytes'] for r in ranks));pinned.append(sum(r['pinned_host_bytes'] for r in ranks))
  else:
   host.append(row['host_rss_bytes'])
   if system=='MoE-Infinity-repaired':
    actual=row['request_ids'];assert len(row['tokens'])==spec['global_requests'] and all(len(t)==64 for t in row['tokens'])
    check_clock(row['release_ns'],row['token_ready_ns'])
    start=row['release_ns'];first=row['token_ready_ns'][0];end=row['token_ready_ns'][-1]
    assert row['cache_before']['resident_bytes']==0 and row['cache_before']['pending_tickets']==0
    assert row['eam_calls']==48*64 and row['eam_candidates']>0
    assert row['cache_after']['priority_evictions']>0
    assert row['cache_after']['peak_accounted_bytes']<=17392730112
    for gpu,budget in enumerate(row['expert_budget_per_gpu']):assert row['cache_after'][f'gpu_{gpu}_peak_charged_bytes']<=budget
   else:
    actual=[r['request_id'] for r in row['requests']]
    assert row['expert_resident_bytes']<=17392730112
    for request in row['requests']:
     assert len(request['tokens'])==len(request['token_ready_us'])==64
     assert request['tokens_evaluated']==spec['input_tokens'] and not request['truncated']
     check_clock(row['release_ns'],[t*1000 for t in request['token_ready_us']])
    start=row['release_ns'];first=max(r['token_ready_us'][0]*1000 for r in row['requests']);end=max(r['token_ready_us'][-1]*1000 for r in row['requests'])
  assert abs(row['TTFT']-(first-start)/1e9)<1e-6 and abs(row['E2E']-(end-start)/1e9)<1e-6,'outer timing disagrees with raw timestamps'
  assert actual==expected,'manifest order/coverage differs'
  rows.append(row)
 if system=='DeepSpeed-ZeRO-Inference':
  for rank in range(4):
   calibration=read(job/f'calibration_rank{rank}.json')
   assert calibration['all_parameter_peak_bytes']<=calibration['budget_bytes']
 metrics={key:summarize([r[key] for r in rows]) for key in ['TTFT','TPOT','E2E','throughput']}
 unstable=any(metrics[k]['spread_percent']>5 for k in ['TTFT','TPOT','E2E'])
 peaks={str(g):0 for g in [0,1,4,5]}
 for line in (job/'resources.jsonl').read_text().splitlines():
  sample=json.loads(line)
  if sample['phase'].get('repeat') in [1,2,3]:
   for gpu in sample['gpus']:peaks[str(gpu['gpu'])]=max(peaks[str(gpu['gpu'])],gpu['used_mib']*2**20)
 return dict(job=str(job),system=system,cell=spec['cell'],status='UNSTABLE' if unstable else 'PASS',metrics=metrics,sampled_peak_hbm_bytes=peaks,host_rss_bytes=max(host),pinned_host_bytes=max(pinned) if pinned else None)
def main(a):
 specs={s['cell']:s for s in read(a.root/'WORKLOADS.json')['cells']};attempts=[]
 for spec in specs.values():check_manifests(spec)
 for job in sorted(a.root.iterdir()):
  if not (job/'result.json').exists():continue
  result=read(job/'result.json')
  if result.get('smoke',False) or result.get('cell') not in specs:continue
  try:attempts.append(audit(job,specs[result['cell']]))
  except Exception as e:attempts.append(dict(job=str(job),system=result.get('system'),cell=result.get('cell'),status='INVALID',error=repr(e)))
 selection=read(a.selection) if a.selection else {};selected=[];missing=[]
 for cell in specs:
  for system in SYSTEMS:
   label=selection.get(cell,{}).get(system)
   matches=[x for x in attempts if x['cell']==cell and x['system']==system and Path(x['job']).name==label]
   if len(matches)==1 and matches[0]['status']=='PASS':selected+=matches
   else:missing.append(dict(cell=cell,system=system,selected_label=label))
 out=dict(status='COMPLETE' if not missing else 'INCOMPLETE',attempts=attempts,selected=selected,missing_or_unstable=missing,notes=['All three repeats retained. No outlier removal.','Spread=(max-min)/mean; CV uses sample standard deviation.','HBM is the observed 1 Hz NVML peak; host RSS is end-of-repeat.','Null pinned memory means unavailable, not zero.','Multi-rank host RSS is summed process RSS and can double-count shared pages; it is not unique physical RAM.'])
 a.output.write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(dict(status=out['status'],audited_attempts=len(attempts),verified_selected=len(selected),remaining=len(missing))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/hwlee/mgo-results/headline_r4_20261007'));p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
