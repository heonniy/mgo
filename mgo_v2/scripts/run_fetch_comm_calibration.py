"""Run only the transport prerequisite and six tiny calibration cells.

Use a fresh --root for a deliberate rerun; a failed gate stops all later work.
"""
import os,subprocess,sys,time,json,argparse,signal
from run_rank_oracle_study import memory,group_rss,process_tree
from pathlib import Path
package=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
root=Path(args.root);root.mkdir(parents=True,exist_ok=False)
for mode in ('T0','T1'):
 for smoke in (True,False):
  label=mode+('_transport' if smoke else '_calibration');out=root/label;out.mkdir(exist_ok=True)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',PYTHONPATH=str(package),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MGO_TRANSPORT=mode)
  for k in ('NCCL_P2P_DISABLE','NCCL_SHM_DISABLE','NCCL_P2P_LEVEL','NCCL_DEBUG','NCCL_DEBUG_SUBSYS','NCCL_DEBUG_FILE'):env.pop(k,None)
  if mode=='T1':env['NCCL_P2P_DISABLE']='1'
  if smoke:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
  cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',str(package/'examples/fetch_comm_calibration.py'),'--output',str(out)]+(['--smoke'] if smoke else [])
  sample=memory()
  assert sample['host_available_bytes']>=512*2**30 and all(g['used_mib']<1024 for g in sample['gpu'].values()),sample
  print('START',label,flush=True)
  started=time.monotonic();samples=[sample];stop_reason=None
  with (out/'run.log').open('w') as f:
   child=subprocess.Popen(cmd,cwd=package,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    time.sleep(2);sample=memory();sample['group_rss_bytes']=group_rss(child.pid);samples.append(sample)
    if time.monotonic()-started>180:stop_reason='180-second calibration timeout'
    if sample['host_available_bytes']<128*2**30 or sample['group_rss_bytes']>32*2**30 or any(g['free_mib']<8192 for g in sample['gpu'].values()):stop_reason='memory guard'
    if stop_reason:
     descendants,_=process_tree(child.pid);os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=10)
     except subprocess.TimeoutExpired:
      for pid in descendants:
       try:os.kill(pid,signal.SIGKILL)
       except ProcessLookupError:pass
      os.killpg(child.pid,signal.SIGKILL);child.wait()
     break
   code=child.returncode
  (out/'status.json').write_text(json.dumps(dict(status='PASS' if code==0 and not stop_reason else 'FAIL',code=code,command=cmd,memory=samples,stop_reason=stop_reason),indent=2)+'\n')
  print('END',label,code,flush=True)
  if code or stop_reason:sys.exit(1)
