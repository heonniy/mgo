"""Four bounded untimed transport checks; fail closed on path mismatch."""
import json,os,signal,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parents[1];root=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003');root.mkdir(exist_ok=True)
packet=P/'experiments/env_e2e_tpot_offload_20261003';receipts=[]
for world,envname in [(8,'env1'),(8,'env2'),(4,'env2'),(4,'env1')]:
 out=root/f'preflight_R{world}_{envname}';out.mkdir(exist_ok=False)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 env=dict(os.environ,PYTHONPATH=str(P),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' if world==8 else '0,1,4,5',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for k in list(env):
  if k.startswith('NCCL_'):del env[k]
 env.update(NCCL_CUMEM_ENABLE='0',NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
 if envname=='env2':env.update(NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1')
 cmd=['/home/hwlee/sub-moe/phase01/.venv/bin/python','-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(P/'examples/env_offload_preflight.py'),'--output',str(out)]
 start=time.time()
 with (out/'run.log').open('w') as f:
  p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=120)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=-1
 lines=[l for path in out.glob('nccl-*.log') for l in path.read_text().splitlines() if 'via ' in l]
 paths=sorted({l.split('via ',1)[1].split()[0] for l in lines})
 valid=code==0 and bool(paths) and (paths==['P2P/IPC'] if envname=='env1' else paths==['SHM/direct/direct'])
 rec=dict(status='PASS' if valid else 'FAIL',world=world,environment=envname,exit_code=code,paths=paths,seconds=time.time()-start,raw_root=str(out))
 receipts.append(rec);(packet/'transport_preflight.json').write_text(json.dumps(receipts,indent=2)+'\n');(packet/f'transport_R{world}_{envname}.log').write_text('\n'.join(lines)+'\n');print(json.dumps(rec),flush=True)
 if not valid:raise RuntimeError('transport preflight failed')
