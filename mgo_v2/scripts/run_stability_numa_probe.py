"""S2: actual visible topology only, no concurrent resource monitor."""
import json,subprocess,os,signal
from run_timing_stability import ROOT,PACKET,P,PYTHON,env_for,sample,safe,write,csvout

def launch(label,worker,args,gpus,preflight=False):
 out=ROOT/label;out.mkdir(exist_ok=False);safe(sample(),True)
 env=env_for('env2');env['CUDA_VISIBLE_DEVICES']=','.join(map(str,gpus))
 if preflight:env.update(NCCL_DEBUG='INFO',NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM',NCCL_DEBUG_FILE=str(out/'nccl-%h-%p.log'))
 cmd=[PYTHON,'-u','-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',str(P/'examples'/worker),'--output',str(out),*args]
 with (out/'run.log').open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=180)
  except BaseException:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
 assert code==0,str(out/'run.log');safe(sample())
 return out

def main():
 # Owner amendment handoff only after the unchanged six-run S1 completes.
 if (ROOT/'amendment_f317328.json').exists():
  import time,sys
  request=json.loads((ROOT/'amendment_f317328.json').read_text())
  parent=os.getppid();assert parent==request['original_driver_pid']
  args=open(f'/proc/{parent}/cmdline','rb').read().split(b'\0')
  assert any(x.endswith(b'/run_timing_stability.py') for x in args)
  deadline=time.monotonic()+900
  while not (ROOT/'amendment_ready').exists():
   if time.monotonic()>deadline:raise TimeoutError('amended continuation not ready')
   time.sleep(1)
  for i,mode in enumerate(['HEAVY','BOUNDARY','BOUNDARY','HEAVY','HEAVY','BOUNDARY'],1):
   assert json.loads((ROOT/f'S1_{i}_{mode}'/'status.json').read_text())['status']=='PASS'
  write(ROOT/'amendment_handoff.json',dict(status='S1_COMPLETE_HANDOFF',old_driver_pid=parent,new_driver_pid=os.getpid(),unix=time.time()))
  os.kill(parent,signal.SIGTERM)  # coordinator only; no science children are active
  os.setsid()
  with (ROOT/'amendment_driver.log').open('w') as log:
   os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
  os.execv(sys.executable,[sys.executable,'-u',str(P/'scripts/run_timing_stability_amended.py')])
 topology=json.loads((ROOT/'topology.json').read_text());assert len(topology['nodes'])==1
 assert all(topology['gpu_numa'][str(g)]['effective_os_node']==0 for g in [0,1])
 h2d=launch('S2_local_H2D','stability_numa_probe.py',['--kind','h2d'],[0])
 preflight=launch('S2_SHM_preflight','env_offload_preflight.py',[],[0,1],True)
 lines=[line for path in preflight.glob('nccl-*.log') for line in path.read_text().splitlines() if 'via ' in line]
 paths=sorted({line.split('via ',1)[1].split()[0] for line in lines});assert paths==['SHM/direct/direct'],paths
 (PACKET/'S2_transport.log').write_text('\n'.join(lines)+'\n')
 shm=launch('S2_same_node_SHM','stability_numa_probe.py',['--kind','shm'],[0,1])
 result=dict(status='PARTIAL_TOPOLOGY_LIMIT',remote_H2D='NOT_AVAILABLE_SINGLE_OS_NODE',cross_numa_SHM='NOT_AVAILABLE_SINGLE_OS_NODE',transport=paths,measurements=[])
 for folder in [h2d,shm]:
  for path in sorted(folder.glob('rank*.json')):result['measurements']+=json.loads(path.read_text())['rows']
 write(PACKET/'S2_results.json',result)
 csvout('numa_table.csv',[dict(kind=r['kind'],gpu=r['gpu'],locality=r['locality'],payload_bytes=r['payload_bytes'],median_ms=r['median_ms'],p90_ms=r['p90_ms']) for r in result['measurements']])
if __name__=='__main__':main()
