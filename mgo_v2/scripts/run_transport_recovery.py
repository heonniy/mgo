#!/usr/bin/env python3
"""One authorized transport attempt per invocation; commit before the next step."""
import argparse,csv,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from run_rank_oracle_study import memory,group_rss,process_tree

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path('/home/hwlee/mgo-results/fetch_comm_pareto_p2p_20261002/recovery')
OUT=PACKAGE/'experiments/fetch_comm_pareto_p2p_20261002'
ENV={
 'R1':{'NCCL_P2P_LEVEL':'LOC'},
 'R2':{'NCCL_P2P_LEVEL':'LOC','NCCL_NET_GDR_LEVEL':'LOC','NCCL_NET_GDR_C2C':'0'},
 'R3':{'NCCL_P2P_LEVEL':'LOC','NCCL_IB_DISABLE':'1'},
 'R3_lo':{'NCCL_P2P_LEVEL':'LOC','NCCL_IB_DISABLE':'1','NCCL_SOCKET_IFNAME':'lo'},
}
def read(p):return json.loads(p.read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--condition',required=True,choices=ENV);parser.add_argument('--calibrate',action='store_true');args=parser.parse_args()
 condition=args.condition;result_path=OUT/'transport_recovery_result.json'
 result=read(result_path) if result_path.exists() else dict(status='RUNNING',plan_commit='92b207049623f4aa85be972d637f5adf7d32c0fc',attempts=[],calibration=None)
 if args.calibrate:
  assert result.get('selected_condition')==condition and result['attempts'][-1]['status']=='PASS'
 else:
  assert not result.get('selected_condition'),'stop at first functional condition'
  order=list(ENV);assert len(result['attempts'])==order.index(condition)
  if condition in ('R2','R3'):
   previous=result['attempts'][-1];assert previous['status']=='FAIL' and 'NET/IB' in previous['selected_transport'] and previous['ib_error'],previous
  if condition=='R3_lo':
   previous=result['attempts'][-1];assert previous['status']=='FAIL' and 'NET/Socket' in previous['selected_transport'] and previous['external_socket_interface'],previous
 target=ROOT/(condition+('_calibration' if args.calibrate else '_smoke'));target.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',PYTHONPATH=str(PACKAGE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MGO_TRANSPORT=condition)
 inherited={k:v for k,v in env.items() if k.startswith('NCCL_')}
 for k in inherited:env.pop(k)
 env.update(ENV[condition])
 if not args.calibrate:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(target/'nccl-%h-%p.log'))
 sample=memory();assert sample['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in sample['gpu'].values()),sample
 worker=PACKAGE/'examples/fetch_comm_calibration.py'
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(worker),'--output',str(target)]+([] if args.calibrate else ['--smoke'])
 state=dict(status='RUNNING',condition=condition,calibration=args.calibrate,command=command,transport_env={k:v for k,v in env.items() if k.startswith('NCCL_')},cleared_inherited_nccl=inherited,worker_sha256=sha(worker),launcher_sha256=sha(Path(__file__)),memory=[sample],started_unix=time.time())
 write(target/'status.json',state);stop_reason=None
 with (target/'run.log').open('w') as log:
  child=subprocess.Popen(command,cwd=PACKAGE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=child.pid
  while child.poll() is None:
   time.sleep(2);sample=memory();sample['group_rss_bytes']=group_rss(child.pid);state['memory'].append(sample)
   if time.time()-state['started_unix']>180:stop_reason='180-second bounded timeout'
   if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>32*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):stop_reason='memory guard'
   write(target/'status.json',state)
   if stop_reason:
    descendants,_=process_tree(child.pid);os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
     for pid in descendants:
      try:os.kill(pid,signal.SIGKILL)
      except ProcessLookupError:pass
     os.killpg(child.pid,signal.SIGKILL);child.wait()
    break
 state.update(status='PASS' if child.returncode==0 and not stop_reason else 'FAIL',exit_code=child.returncode,stop_reason=stop_reason,finished_unix=time.time());write(target/'status.json',state)
 if args.calibrate:
  result['calibration']=dict(condition=condition,status=state['status'],raw_root=str(target));result['status']='CALIBRATION_READY' if state['status']=='PASS' else 'CALIBRATION_FAILED'
 else:
  files=sorted(target.glob('nccl-*.log'));text='\n'.join(p.read_text() for p in files);run=(target/'run.log').read_text()
  channels=[line for line in text.splitlines() if ' via ' in line]
  selected=sorted({line.split(' via ',1)[1].strip() for line in channels})
  p2p=any('P2P/' in line or 'NVLS' in line for line in channels)
  receipts=[]
  for line in run.splitlines():
   if line.startswith('{'):
    try:receipts.append(json.loads(line))
    except json.JSONDecodeError:pass
  payload=sorted(r['rank'] for r in receipts if r.get('status')=='PASS')==[0,1,2,3]
  passed=state['status']=='PASS' and payload and len(files)==4 and bool(selected) and not p2p
  attempt=dict(condition=condition,status='PASS' if passed else 'FAIL',p2p_used=p2p,selected_transport='; '.join(selected),payload_validation=payload,
   ib_error=('IBV_WC_' in text or 'NET/IB' in text and any(s in text+run for s in ('ncclRemoteError','connection error'))),
   external_socket_interface=('NET/Socket' in text and 'NCCL_SOCKET_IFNAME=lo' not in text and '192.168.' in text),
   error='' if passed else ('IBV_WC_RETRY_EXC_ERR' if 'IBV_WC_RETRY_EXC_ERR' in text else stop_reason or 'transport/payload gate failed'),
   topology_lines=sorted({line.split('NCCL INFO ',1)[-1] for line in text.splitlines() if 'nNodes ' in line}),
   environment=ENV[condition],raw_root=str(target),peer_median_ms=None,h2d_median_ms=None)
  result['attempts'].append(attempt)
  result['status']='FUNCTIONAL_NON_P2P' if passed else ('BLOCKED_SYNTHETIC_TRANSPORT' if condition=='R3_lo' else 'RECOVERY_IN_PROGRESS')
  if passed:result['selected_condition']=condition
  logs=OUT/'transport_recovery_logs'/condition;logs.mkdir(parents=True,exist_ok=False)
  for file in files+[target/'run.log',target/'status.json']:shutil.copyfile(file,logs/file.name)
 write(result_path,result)
 fields=['condition','status','p2p_used','selected_transport','error','peer_median_ms','h2d_median_ms','notes']
 with (OUT/'transport_recovery.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader()
  for a in result['attempts']:w.writerow(dict(a,notes='Exact condition from TRANSPORT_RECOVERY.md; no model loaded.'))
 print(json.dumps(dict(status=result['status'],condition=condition,stage_status=state['status'])),flush=True)
 if state['status']!='PASS':sys.exit(1)
if __name__=='__main__':main()
