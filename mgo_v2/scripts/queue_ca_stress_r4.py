"""Wait for committed R8 completion, then run exactly two owner-fixed GPU sets."""
import os,json,time,subprocess,csv,signal
from pathlib import Path
P=Path(__file__).resolve().parents[1];ROOT=Path('/home/hwlee/mgo-results/ca_stress_physical_validation_20261004');PACKET=P/'experiments/ca_stress_physical_validation_20261004'
PYTHON='/home/hwlee/sub-moe/phase01/.venv/bin/python'
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def process_identity(pid):
 try:
  value=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
  return None if value[0]=='Z' else value[19]
 except FileNotFoundError:return None

def comparison():
 rows=[]
 for name,path in [('R8_best',PACKET),('R4_0146',PACKET/'R4_0146'),('R4_0123',PACKET/'R4_0123')]:
  for r in csv.DictReader((path/'comparisons.csv').open()):rows.append(dict(configuration=name,**r))
 with (PACKET/'R4_R8_comparison.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# Two fixed R4 GPU sets and the R8 stress reference','','All are optimized communication-stress workloads, not dataset averages.','R4 uses sample205/DP4/BR73, 32 requests; R8 uses sample81/DP86/BR42,','64 requests. The R4/R8 contrast changes workload as well as rank count and','cannot isolate a rank-count effect. The two R4 sets preserve request/rank','membership, seed, CPU affinity and all other settings.','','| Configuration | Env | Primary comparison valid | E2E gain BR→CA | TPOT gain | Physical peer reduction |','|---|---|---|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['configuration']} | {r['environment']} | {r.get('primary_comparison_valid','unknown')} | {float(r['E2E_wall_gain']):.2%} | {float(r['TPOT_gain']):.2%} | {float(r['peer_reduction']):.2%} |")
 for name,path in [('R8_best',PACKET),('R4_0146',PACKET/'R4_0146'),('R4_0123',PACKET/'R4_0123')]:
  v=json.loads((path/'validation.json').read_text());lines+=['',f"{name}: timing_stable={v['timing_stable']}, timed_generations={v['timed_generations']}."]
 lines+=['','See each configuration RESULTS.md, timing_summary.csv and counter_summary.csv','for full ranges, stability and CPU/physical differences. No further sweep.']
 (PACKET/'R4_RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
 origin=json.loads((ROOT/'process.json').read_text());pid=origin['pid'];identity=process_identity(pid)
 state=dict(status='WAITING_FOR_R8',R8_pid=pid,R8_process_identity=identity,owner_commit='a121c85dc6820a36b03854a9c51e2109f89aca0c',physical_gpu_sets=[[0,1,4,6],[0,1,2,3]],started_unix=time.time())
 write(ROOT/'R4_queue_status.json',state)
 while identity is not None and process_identity(pid)==identity:
  if (ROOT/'STOP_R4_QUEUE').exists():state['status']='OWNER_STOPPED';write(ROOT/'R4_queue_status.json',state);return
  time.sleep(15)
 current=json.loads((PACKET/'status.json').read_text())
 if current['status']!='COMPLETE':
  state.update(status='R8_REQUIRES_ATTENTION',R8_status=current);write(ROOT/'R4_queue_status.json',state);return
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(PACKET/'status.json')],cwd=P.parent,check=True)
 child=None
 try:
  for variant in ['0146','0123']:
   if (ROOT/'STOP_R4_QUEUE').exists():raise RuntimeError('Owner stopped R4 queue')
   state.update(status='RUNNING_R4',variant=variant);write(ROOT/'R4_queue_status.json',state)
   env=dict(os.environ,MGO_R4_GPU_SET=variant,MGO_R4_CHAIN='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{P}:{P}/scripts')
   with (ROOT/('R4_'+variant+'_driver.log')).open('w') as log:
    child=subprocess.Popen([PYTHON,'-u',str(P/'scripts/run_ca_stress_r4.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=child.pid;write(ROOT/'R4_queue_status.json',state)
    while child.poll() is None:
     if (ROOT/'STOP_R4_QUEUE').exists():
      # Ask the guarded driver to stop; never kill its parent and orphan
      # a torchrun group. A timed run exits at its next safe boundary.
      (ROOT/('R4_'+variant)/'STOP').touch()
     time.sleep(15)
    assert child.returncode==0,f'R4 {variant} failed; receipts retained'
  comparison();state.update(status='COMPLETE',finished_unix=time.time())
 except BaseException as exc:state.update(status='FAILED_OR_STOPPED',error=repr(exc),finished_unix=time.time());raise
 finally:
  # Each child waits/reaps all scientific torchrun processes before returning.
  from run_timing_stability import restore,commit
  state['resident_models']=restore();write(ROOT/'R4_queue_status.json',state)
  write(PACKET/'R4_queue_completion.json',state)
  import run_timing_stability as harness
  harness.PACKET=PACKET;commit('results: finish two fixed R4 GPU sets and restore model workers')
if __name__=='__main__':main()
