"""Bounded owner diagnostic: no automatic resumption of policy timing."""
import json,os,signal,statistics,subprocess,time,hashlib,csv
from pathlib import Path
from run_env_offload_cell import snapshot,write,P,process_tree
ROOT=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004')
OLD=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003')
PLAN=OLD/'P_CA-rep_env1_PLAN_0'
PACKET=P/'experiments/timing_stability_numa_20261004'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'

def env_for(environment):
 env=dict(os.environ,PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/examples:{P}/scripts',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2')
 for k in list(env):
  if k.startswith('NCCL_'):del env[k]
 env['NCCL_CUMEM_ENABLE']='0'
 if environment=='env2':env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
 cache=OLD/'compile_cache'/f'P_CA-rep_{environment}'
 assert cache.is_dir()
 env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'))
 return env

def sample(pid=None,heavy=False):
 started=time.monotonic();row=snapshot();row['unix']=time.time()
 if pid:
  pids,rss=process_tree(pid);row['rss_bytes']=rss
  if heavy:
   pss=0
   for child in pids:
    try:pss+=next(int(l.split()[1])*1024 for l in Path(f'/proc/{child}/smaps_rollup').read_text().splitlines() if l.startswith('Pss:'))
    except (FileNotFoundError,ProcessLookupError):pass
   row['pss_bytes']=pss
  # Refresh the set after a long PSS scan, as the historical guard does.
  pids,_=process_tree(pid)
 else:pids=set()
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
 row['foreign_pids']=[int(x) for x in apps.splitlines() if int(x) not in pids and int(x)!=pid]
 row['collection_seconds']=time.monotonic()-started
 return row

def safe(row,initial=False):
 assert row['host_available_bytes']>=(768 if initial else 256)*2**30,'host memory guard'
 assert row.get('pss_bytes',0)<=768*2**30,'PSS guard'
 assert not row['foreign_pids'],'foreign compute process'
 assert all(g['free_mib']>(76000 if initial else 8192) and g['temperature_c']<(65 if initial else 85) for g in row['gpus']),'GPU memory/temperature guard'
 assert not (ROOT/'STOP').exists(),'owner STOP'

def commit(message):
 subprocess.run(['git','add',str(PACKET)],cwd=P.parent,check=True)
 if subprocess.run(['git','diff','--cached','--quiet'],cwd=P.parent).returncode:
  subprocess.run(['git','commit','-m',message],cwd=P.parent,check=True)
  pushed=subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'],cwd=P.parent)
  if pushed.returncode:
   # Owner may publish the next plan while a bounded phase is running.
   # Preserve both histories; never force-push or discard either side.
   subprocess.run(['git','fetch','origin','codex/mgo-r4-trajectory-results-20261002'],cwd=P.parent,check=True)
   merged=subprocess.run(['git','merge','--no-edit','FETCH_HEAD'],cwd=P.parent)
   if merged.returncode:
    subprocess.run(['git','merge','--abort'],cwd=P.parent,check=True)
    raise RuntimeError('Publication merge conflict; scientific receipts retained')
   subprocess.run(['git','push','origin','HEAD:codex/mgo-r4-trajectory-results-20261002'],cwd=P.parent,check=True)

def run(label,mode,affinity,environment,horizon,residency=None):
 out=ROOT/label
 if (out/'status.json').exists():
  state=json.loads((out/'status.json').read_text());assert state['status']=='PASS',str(out);return state
 out.mkdir(exist_ok=False);initial=sample();safe(initial,True)
 worker='timing_stability_worker.py' if residency is None else 'timing_stability_residency_worker.py'
 cmd=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(P/'examples'/worker),'--output',str(out),'--plan',str(PLAN),'--horizon',str(horizon),'--mode',mode,'--affinity',affinity,'--environment',environment]
 if residency is not None:cmd+=['--residency',residency]
 state=dict(residency=residency,status='RUNNING',label=label,mode=mode,affinity=affinity,environment=environment,horizon=horizon,started_unix=time.time(),command=cmd,initial=initial,samples=[],source_sha256=hashlib.sha256((P/'examples'/worker).read_bytes()).hexdigest())
 reason=None;timed=False
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env_for(environment),stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;write(out/'status.json',state)
  try:
   while proc.poll() is None:
    if not timed and len(list(out.glob('ready*.json')))==8:
     # This final resource scan completes before any rank is released.
     before=sample(proc.pid,True);safe(before);state['before_measure']=before;state['GO_unix']=time.time();write(out/'status.json',state);(out/'GO').touch();timed=True
     if mode=='BOUNDARY':
      # No procfs walk, ps, nvidia-smi, or resource polling until all ranks exit.
      proc.wait(timeout=3600)
      state['boundary_wait_finished_unix']=time.time()
      break
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    if proc.poll() is not None:break
    row=sample(proc.pid,True);state['samples'].append(row);write(out/'status.json',state);safe(row)
    if time.time()-state['started_unix']>7200:raise TimeoutError('bounded phase timeout')
  except BaseException as exc:
   reason=repr(exc)
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  state.update(exit_code=proc.returncode,finished_unix=time.time(),error=reason)
 try:
  after=sample();state['after_measure']=after;safe(after)
  assert proc.returncode==0 and reason is None and timed
  rows=[json.loads((out/f'rank{i}.json').read_text()) for i in range(8)]
  assert all(r['status']=='PASS' and r['no_compile_in_measure'] for r in rows)
  state.update(status='PASS',**{k:max(r[k] for r in rows) for k in ['E2E_wall','decode_wall','TPOT']},rank_receipts=rows)
 except BaseException as exc:state.update(status='FAIL',validation_error=repr(exc))
 write(out/'status.json',state)
 compact={k:v for k,v in state.items() if k!='samples'}
 compact['monitor_scan_seconds']=[x['collection_seconds'] for x in state['samples']]
 (PACKET/'receipts').mkdir(exist_ok=True);write(PACKET/'receipts'/f'{label}.json',compact)
 commit(f'results: timing stability {label} {state["status"]}')
 assert state['status']=='PASS',str(out/'run.log')
 return state

def spread(rows,key='decode_wall'):
 values=[r[key] for r in rows];median=statistics.median(values)
 return dict(n=len(values),median=median,min=min(values),max=max(values),spread_percent=100*(max(values)-min(values))/median)

def csvout(name,rows):
 with (PACKET/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def restore():
 # Driver is the only science parent; each bounded run was waited/reaped.
 from batch_comm_common import start_idle_load,LOAD,owned
 if (ROOT/'STOP').exists():return dict(status='SKIPPED_OWNER_STOP')
 ps=start_idle_load();deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  verified=[]
  for p in ps:
   path=LOAD/f'gpu{p["gpu"]}.json'
   if path.exists():
    row=json.loads(path.read_text())
    if row.get('pid')==p['pid'] and owned(p['pid']) and row.get('iterations',0)>=2 and time.time()-row.get('unix',0)<90:verified.append(p)
  if len(verified)==len(ps):break
  time.sleep(15)
 return dict(processes=ps,verified=verified,all_eight_verified=len(verified)==8)

def main():
 outcome=dict(status='RUNNING',automatic_policy_resume=False)
 write(PACKET/'status.json',outcome)
 try:
  assert json.loads((ROOT/'prefix_validation.json').read_text())['status']=='PASS'
  s1=[run(f'S1_{i}_{mode}',mode,'original','env1',64) for i,mode in enumerate(['HEAVY','BOUNDARY','BOUNDARY','HEAVY','HEAVY','BOUNDARY'],1)]
  heavy=spread([r for r in s1 if r['mode']=='HEAVY']);boundary=spread([r for r in s1 if r['mode']=='BOUNDARY'])
  confound=boundary['spread_percent']<=5 and (heavy['median']>=1.1*boundary['median'] or heavy['spread_percent']>=2*boundary['spread_percent'])
  write(PACKET/'S1_summary.json',dict(HEAVY=heavy,BOUNDARY=boundary,MONITOR_CONFOUND=confound,attribution='UNRESOLVED' if boundary['spread_percent']>5 else 'GATE_PASSED' if confound else 'NOT_ESTABLISHED'))
  csvout('monitor_table.csv',[{k:r[k] for k in ['label','mode','E2E_wall','decode_wall','TPOT']} for r in s1]);commit('results: monitor interference gate')
  subprocess.run([PYTHON,str(P/'scripts/run_stability_numa_probe.py')],cwd=P.parent,env=env_for('env2'),check=True)
  commit('results: visible NUMA domain and bounded local probes')
  s3=[run(f'S3_{i}_{env}','BOUNDARY','fixed',env,64) for i,env in enumerate(['env1','env2','env2','env1','env1','env2'],1)]
  checks={env:spread([r for r in s3 if r['environment']==env]) for env in ['env1','env2']}
  write(PACKET/'S3_summary.json',checks);commit('results: fixed affinity decode64 stability gate')
  stable=all(v['spread_percent']<=5 for v in checks.values())
  outcome.update(status='HARNESS_UNSTABLE',decode64=checks)
  if stable:
   confirmation=[run(f'S3_confirm_{i}_{env}','BOUNDARY','fixed',env,256) for i,env in enumerate(['env1','env2','env2','env1','env1','env2'],1)]
   full={env:spread([r for r in confirmation if r['environment']==env]) for env in ['env1','env2']}
   outcome.update(status='HARNESS_STABLE' if all(v['spread_percent']<=5 for v in full.values()) else 'HARNESS_UNSTABLE_256',decode256=full)
  outcome['S2_limitation']='Only one OS NUMA domain is exposed; remote-H2D and cross-NUMA SHM not measurable. No physical locality inference.'
 except BaseException as exc:
  outcome.update(status='DIAGNOSTIC_FAILED_OR_STOPPED',error=repr(exc));raise
 finally:
  write(PACKET/'status.json',outcome)
  rows=[]
  for p in (PACKET/'receipts').glob('*.json'):
   r=json.loads(p.read_text())
   if r['status']=='PASS':rows.append({k:r[k] for k in ['label','mode','affinity','environment','horizon','E2E_wall','decode_wall','TPOT']})
  if rows:csvout('timing_samples.csv',rows)
  (PACKET/'RESULTS.md').write_text('# '+outcome['status']+'\n\n'+json.dumps(outcome,indent=2)+'\n\nSee timing_samples.csv, per-rank receipts, S1_summary.json and S3_summary.json\nwhen available. No policy ranking or automatic old-matrix resume.\n')
  outcome['resident_model_handoff']=restore();write(PACKET/'status.json',outcome)
  commit('results: finish bounded timing-stability diagnostic')
if __name__=='__main__':main()
