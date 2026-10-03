"""f317328 continuation after unchanged S1: residency, H2D scaling, all SHM pairs."""
import json,os,signal,statistics,subprocess,time,itertools
from pathlib import Path
from run_timing_stability import ROOT,PACKET,P,PYTHON,env_for,sample,safe,write,run,spread,commit,csvout,restore

def launch(label,args,world,preflight=False):
 out=ROOT/label;out.mkdir(exist_ok=False);initial=sample();safe(initial,True)
 env=env_for('env2');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(world)))
 worker='env_offload_preflight.py' if preflight else 'stability_transport_scaling.py'
 if preflight:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
 command=[PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples'/worker),'--output',str(out),*args]
 state=dict(status='RUNNING',command=command,initial=initial,started_unix=time.time(),concurrent_monitor=False)
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  try:code=proc.wait(timeout=600)
  except BaseException as exc:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   state.update(status='FAIL',error=repr(exc),finished_unix=time.time());write(out/'status.json',state);raise
 state.update(status='PASS' if code==0 else 'FAIL',exit_code=code,finished_unix=time.time(),after=sample());write(out/'status.json',state);safe(state['after']);assert code==0,str(out/'run.log')
 ranks=[json.loads((out/f'rank{i}.json').read_text()) for i in range(world)];assert all(r['status']=='PASS' for r in ranks)
 return out,ranks,state

def residency_gate(rows):
 groups={mode:[r for r in rows if r['residency']==mode] for mode in ['NORMAL','PRETOUCH']}
 result={mode:dict(decode=spread(rs),minor_faults=[sum(x['page_faults']['minor_delta'] for x in r['rank_receipts']) for r in rs],major_faults=[sum(x['page_faults']['major_delta'] for x in r['rank_receipts']) for r in rs]) for mode,rs in groups.items()}
 a,b=result['NORMAL'],result['PRETOUCH'];minor_a=statistics.median(a['minor_faults']);minor_b=statistics.median(b['minor_faults']);major_a=statistics.median(a['major_faults']);major_b=statistics.median(b['major_faults'])
 faults=(minor_a>0 and minor_b<=.5*minor_a and major_b<=major_a) or (major_a>0 and major_b<=.5*major_a and minor_b<=minor_a)
 variability=a['decode']['spread_percent']>5 and b['decode']['spread_percent']<=5 and b['decode']['spread_percent']<=.5*a['decode']['spread_percent']
 confound=bool(faults and variability and b['decode']['median']<=a['decode']['median'])
 result.update(PAGE_RESIDENCY_CONFOUND=confound,S3_residency='PRETOUCH' if confound else 'NORMAL',fault_scope='whole rank process, not expert-only; two samples per mode',NORMAL_already_touches_full_pool=True,disk_IO_attribution='NOT_ESTABLISHED',criterion='NORMAL spread >5%; PRETOUCH <=5% and <=half NORMAL spread; median no worse; >=50% reduction in nonzero median minor or major faults without increase in the other category')
 return result

def transport_stages():
 h2d=[];receipts=[]
 for world in [1,2,4,8]:
  out,ranks,state=launch(f'S2B_H2D_{world}',['--kind','h2d'],world);receipts.append(state)
  h2d.extend(r['rows'][0] for r in ranks)
 baseline=next(r for r in h2d if r['concurrency']==1)
 for row in h2d:row['latency_slowdown_vs_single_GPU0']=row['median_ms']/baseline['median_ms'];row['bandwidth_ratio_vs_single_GPU0']=row['median_GBps']/baseline['median_GBps']
 write(PACKET/'S2B_H2D.json',dict(status='PASS',baseline_gpu=0,baseline_limitation='Nested GPU-ID groups, not independent single-GPU baselines for every device',aggregate_definition='total bytes divided by latest host completion minus earliest host start, including observed start skew',rows=h2d,receipts=receipts))
 csvout('H2D_scaling.csv',[{k:r[k] for k in ['concurrency','gpu','median_ms','p90_ms','median_GBps','p90_GBps','aggregate_wall_median_GBps','aggregate_wall_p90_GBps','latency_slowdown_vs_single_GPU0']} for r in h2d]);commit('results: concurrent 1 2 4 8 GPU H2D scaling')
 out,ranks,state=launch('S2C_SHM_preflight',[],8,True)
 lines=[line for path in out.glob('nccl-*.log') for line in path.read_text().splitlines() if 'via ' in line];paths=sorted({line.split('via ',1)[1].split()[0] for line in lines});assert paths==['SHM/direct/direct'],paths
 (PACKET/'S2C_transport.log').write_text('\n'.join(lines)+'\n')
 out,ranks,state=launch('S2C_SHM_pairs',['--kind','shm'],8)
 pairs=[row for rank in ranks for row in rank['rows'] if row['initiator']];assert len(pairs)==28*3
 assert {(tuple(row['pair']),row['payload_bytes']) for row in pairs}=={(pair,kib*1024) for pair in itertools.combinations(range(8),2) for kib in [32,128,512]}
 ratios=[]
 for kib in [32,128,512]:
  values=[r['median_ms'] for r in pairs if r['payload_bytes']==kib*1024];ratios.append(dict(payload_kib=kib,min_pair_median_ms=min(values),max_pair_median_ms=max(values),max_min_pair_ratio=max(values)/min(values)))
 write(PACKET/'S2C_SHM.json',dict(status='PASS',transport=paths,pair_order='lexicographic unordered GPU-ID pairs; low-ID initiates dispatch-return',no_NUMA_labels=True,rows=pairs,all_rank_results=ranks,ratios=ratios,receipt=state))
 csvout('SHM_pair_matrix.csv',[dict(gpu_a=r['pair'][0],gpu_b=r['pair'][1],payload_bytes=r['payload_bytes'],median_ms=r['median_ms'],p90_ms=r['p90_ms']) for r in pairs]);csvout('SHM_pair_ratios.csv',ratios);commit('results: empirical Env2 SHM matrix across all 28 GPU pairs')

def report(outcome):
 rows=[]
 for path in sorted((PACKET/'receipts').glob('*.json')):
  row=json.loads(path.read_text())
  if row['status']=='PASS':rows.append(row)
 samples=[]
 for row in rows:
  samples.append(dict(label=row['label'],mode=row['mode'],affinity=row['affinity'],residency=row.get('residency') or 'BASELINE',environment=row['environment'],horizon=row['horizon'],E2E_wall=row['E2E_wall'],decode_wall=row['decode_wall'],TPOT=row['TPOT'],monitor_scan_median_seconds=statistics.median(row.get('monitor_scan_seconds',[0])) if row.get('monitor_scan_seconds') else 0))
 if samples:csvout('timing_samples.csv',samples)
 stability=[]
 for stage in ['decode64','decode256']:
  for env,v in outcome.get(stage,{}).items():stability.append(dict(stage=stage,environment=env,affinity='fixed_disjoint_24_vCPUs_per_rank',residency=outcome.get('S3_residency','NORMAL'),**v,status='PASS' if v['spread_percent']<=5 else 'FAIL'))
 if stability:csvout('stability_table.csv',stability)
 lines=['# '+outcome['status'],'','Policy timing remains paused. No automatic resume.','Guest-visible topology does not identify physical host NUMA.','']
 s1=json.loads((PACKET/'S1_summary.json').read_text());lines+=['## S1 monitor comparison','','| Mode | n | Decode median [min, max], s | Spread |','|---|---:|---:|---:|']
 for mode in ['HEAVY','BOUNDARY']:
  r=s1[mode];lines.append(f"| {mode} | {r['n']} | {r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}] | {r['spread_percent']:.2f}% |")
 lines+=['',f"MONITOR_CONFOUND: **{s1['MONITOR_CONFOUND']}**.",'']
 if (PACKET/'S2A_summary.json').exists():
  a=json.loads((PACKET/'S2A_summary.json').read_text());lines+=['## S2A residency','','NORMAL already reads the full CPU expert pool during load and performs a','complete warmup. PRETOUCH adds CPU reads immediately before measurement.','Two repeats per mode are descriptive; fault counts cover whole rank processes.','','| Mode | Decode median [min, max], s | Spread | Minor faults, totals per run | Major faults, totals per run |','|---|---:|---:|---|---|']
  for mode in ['NORMAL','PRETOUCH']:
   r=a[mode];v=r['decode'];lines.append(f"| {mode} | {v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}] | {v['spread_percent']:.2f}% | {r['minor_faults']} | {r['major_faults']} |")
  lines+=['',f"PAGE_RESIDENCY_CONFOUND: **{a['PAGE_RESIDENCY_CONFOUND']}**; S3 uses **{a['S3_residency']}**.",'No disk-I/O attribution is made solely from file backing or minor faults.','']
 for fname,title in [('S2B_H2D.json','S2B H2D scaling'),('S2C_SHM.json','S2C empirical SHM pairs')]:
  if (PACKET/fname).exists():lines+=['## '+title,'',f'See `{fname}` and the corresponding CSV for full per-device/pair distributions.','']
 if stability:
  lines+=['## Fixed-affinity stability','','| Horizon | Env | n | Decode median [min, max], s | Spread | Gate |','|---|---|---:|---:|---:|---|']
  for r in stability:lines.append(f"| {r['stage']} | {r['environment']} | {r['n']} | {r['median']:.3f} [{r['min']:.3f}, {r['max']:.3f}] | {r['spread_percent']:.2f}% | {r['status']} |")
 lines+=['','All accepted model samples pass per-rank PLAN-prefix token/cache hashes,','frozen route checks and no-compilation checks. Fault/resource reads occur','outside globally synchronized timing boundaries. Full receipts retain hashes,','affinity and safety records. H6 remains unverified; H7/H8 are screened only to','the limits of their existing logs. No policy ranking follows from this packet.']
 (PACKET/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
 outcome=dict(status='RUNNING_AMENDED',stage='S2A',owner_commit='f317328e5b9d5b1ae89344604f7272dba2323872',automatic_policy_resume=False)
 write(PACKET/'status.json',outcome)
 handoff=json.loads((ROOT/'amendment_handoff.json').read_text());write(PACKET/'amendment_handoff.json',handoff);commit('ops: hand off completed S1 to amended diagnostic')
 try:
  rows=[run(f'S2A_{i}_{mode}','BOUNDARY','original','env1',64,residency=mode) for i,mode in enumerate(['NORMAL','PRETOUCH','PRETOUCH','NORMAL'],1)]
  residency=residency_gate(rows);write(PACKET/'S2A_summary.json',residency);commit('results: frozen NORMAL PRETOUCH page-residency diagnostic')
  outcome.update(stage='S2B_S2C',S3_residency=residency['S3_residency']);write(PACKET/'status.json',outcome);transport_stages()
  outcome['stage']='S3_decode64';write(PACKET/'status.json',outcome)
  rows=[run(f'S3_{i}_{env}','BOUNDARY','fixed',env,64,residency=residency['S3_residency']) for i,env in enumerate(['env1','env2','env2','env1','env1','env2'],1)]
  checks={env:spread([r for r in rows if r['environment']==env]) for env in ['env1','env2']};write(PACKET/'S3_summary.json',checks);commit('results: amended fixed-affinity decode64 stability gate')
  outcome.update(status='HARNESS_UNSTABLE',decode64=checks)
  if all(v['spread_percent']<=5 for v in checks.values()):
   outcome['stage']='S3_decode256';write(PACKET/'status.json',outcome)
   rows=[run(f'S3_confirm_{i}_{env}','BOUNDARY','fixed',env,256,residency=residency['S3_residency']) for i,env in enumerate(['env1','env2','env2','env1','env1','env2'],1)]
   full={env:spread([r for r in rows if r['environment']==env]) for env in ['env1','env2']}
   outcome.update(status='HARNESS_STABLE' if all(v['spread_percent']<=5 for v in full.values()) else 'HARNESS_UNSTABLE_256',decode256=full)
  outcome['stage']='FINISHED'
 except BaseException as exc:outcome.update(status='DIAGNOSTIC_FAILED_OR_STOPPED',error=repr(exc));raise
 finally:
  write(PACKET/'status.json',outcome)
  try:report(outcome)
  finally:
   outcome['resident_model_handoff']=restore();write(PACKET/'status.json',outcome);commit('results: finish amended timing stability diagnostic and GPU handoff')
if __name__=='__main__':main()
