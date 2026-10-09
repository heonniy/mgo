"""Exclusive GPU job supervisor with memory guards and resource receipts."""
import argparse,os,time,subprocess,signal,json,hashlib
from pathlib import Path
import run_full_pinned_r4 as c
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
def wait_for_gpu_release(seconds=30):
 deadline=time.monotonic()+seconds
 observed=[]
 while True:
  occupants=c.foreign_on_targets()
  if not occupants or time.monotonic()>=deadline:return occupants,observed
  observed.append(dict(unix=time.time(),occupants=occupants));time.sleep(1)
def main(a):
 llama_workers=('headline_llama_sync_worker.py','headline_llama_deepseek_sync_worker.py')
 if a.ours_final:
  assert a.worker=='headline_ours_worker.py' and a.ranks==4
  assert a.expert_executor in (None,'native') and a.policy in (None,'LA_CA_NEAR')
  assert not a.legacy_decode_layout and not a.capture_eviction_trace
  a.expert_executor='native';a.native_prefill=True;a.prefetch_off=True
  a.prefill_optimized=True;a.prefill_layout_fast=True;a.decode_layout_fast=True
  a.policy='LA_CA_NEAR'
 assert not (a.decode_layout_fast and a.legacy_decode_layout),'choose one decode layout'
 requested_timeout=a.timeout
 if a.worker=='headline_llama_worker.py' and a.cell=='R4_C30_B64_L512_O64' and not a.smoke:
  a.timeout=max(a.timeout,14400)
 out=ROOT/a.label;assert not out.exists(),'new label required to preserve attempts';assert c.host_available()>=384*2**30
 out.mkdir(parents=True);state=dict(status='RUNNING',started=time.time(),system=a.system,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=c.P.parent,text=True).strip());stopped=[];proc=None
 if a.workloads:
  workload_path=Path(a.workloads).resolve()
  state.update(workload_manifest=str(workload_path),workload_manifest_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest())
 state.update(requested_timeout_seconds=requested_timeout,effective_timeout_seconds=a.timeout,
              nccl_p2p_disable=bool(a.nccl_p2p_disable),
              nccl_ib_disable=bool(a.nccl_p2p_disable))
 min_gpu_free_mib=int(os.environ.get('MGO_MIN_GPU_FREE_MIB','0'))
 assert min_gpu_free_mib>=0
 state['min_gpu_free_mib']=min_gpu_free_mib
 try:
  stopped=c.stop_target_idle()
  occupants,observed=wait_for_gpu_release()
  state['preflight_release_observations']=observed
  assert not occupants,f'GPU occupants remain after bounded cleanup wait: {occupants}'
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,4,5',MGO_V2_PHYSICAL_GPUS='0,1,4,5',OMP_NUM_THREADS='2',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TORCHINDUCTOR_COMPILE_THREADS='2',PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{c.P}:{c.P/"scripts"}:{c.P/"examples"}')
  if a.ours_final or a.expert_executor=='native' or a.worker=='headline_ours_deepseek_worker.py':
   native_build=Path('/home/hwlee/mgo-tools/native-expert-build/bin')
   assert (native_build/'ninja').is_file(),'native executor requires its audited ninja build tool'
   env['PATH']=str(native_build)+os.pathsep+env['PATH']
   env['CUDA_HOME']='/usr/local/cuda'
   env['TORCH_CUDA_ARCH_LIST']='9.0'
   env['MAX_JOBS']='1'
  for k in list(env):
   if k.startswith('NCCL_'):del env[k]
  env['NCCL_CUMEM_ENABLE']='0';env['PYTHONFAULTHANDLER']='1'
  if a.worker=='headline_ours_deepseek_worker.py':
   env['NUMBA_CACHE_DIR']='/tmp/mgo-main-table-deepseek-numba-v1'
  if a.nccl_p2p_disable:
   # With P2P off, this host selects a failing NET/IB path. Keep the
   # established same-host env2 transport on SHM by disabling IB as well.
   env['NCCL_P2P_DISABLE']='1';env['NCCL_IB_DISABLE']='1'
  if a.workloads:env['MGO_HEADLINE_WORKLOADS']=str(Path(a.workloads).resolve())
  command=[a.python,'-u']
  if a.ranks>1:command+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.ranks}']
  command +=[str(c.P/'examples'/a.worker),'--cell',a.cell,'--output',str(out)]+(['--smoke'] if a.smoke else [])
  if a.repeats is not None:command+=['--repeats',str(a.repeats)]
  if a.expert_executor is not None:
   assert a.worker=='headline_ours_worker.py';command+=['--expert-executor',a.expert_executor]
  if a.native_prefill:
   assert a.worker=='headline_ours_worker.py' and a.expert_executor=='native' and a.prefill_optimized
   command+=['--native-prefill']
  if a.prefetch_off:
   assert a.worker=='headline_ours_worker.py';command+=['--prefetch-off']
  if a.h2d_serial_ablation:
   assert a.worker=='headline_ours_worker.py' and a.expert_executor=='native' and a.prefetch_off and a.decode_layout_fast
   command+=['--h2d-serial-ablation']
  if a.llama_threads is not None:
   assert a.worker in llama_workers and a.ranks==1
   command+=['--threads',str(a.llama_threads)]
  elif a.worker in llama_workers:
   raise AssertionError('synchronous llama jobs must explicitly declare --llama-threads')
  if a.llama_cuda_graphs is not None:
   assert a.worker in llama_workers and a.ranks==1
   command+=['--cuda-graphs',a.llama_cuda_graphs]
  elif a.worker in llama_workers:
   raise AssertionError('synchronous llama jobs must explicitly declare --llama-cuda-graphs')
  if a.llama_graph_reuse is not None:
   assert a.worker in llama_workers and a.ranks==1
   command+=['--graph-reuse',a.llama_graph_reuse]
  elif a.worker in llama_workers:
   raise AssertionError('synchronous llama jobs must explicitly declare --llama-graph-reuse')
  if a.llama_expert_placement is not None:
   assert a.worker in llama_workers and a.ranks==1
   command+=['--expert-placement',a.llama_expert_placement]
  if a.prefill_optimized:
   assert a.worker=='headline_ours_worker.py';command+=['--prefill-optimized']
  if a.prefill_diagnostic:
   assert a.worker=='headline_ours_worker.py' and a.prefill_optimized;command+=['--prefill-diagnostic']
  if a.prefill_layout_fast:
   assert a.worker=='headline_ours_worker.py' and a.prefill_optimized;command+=['--prefill-layout-fast']
  if a.post_prefill_diagnostic:
   assert a.worker=='headline_ours_worker.py' and a.prefill_layout_fast;command+=['--post-prefill-diagnostic']
  if a.legacy_decode_layout:
   assert a.worker=='headline_ours_worker.py';command+=['--legacy-decode-layout']
  if a.decode_layout_fast:
   assert a.worker=='headline_ours_worker.py';command+=['--decode-layout-fast']
  if a.policy:
   assert a.worker=='headline_ours_worker.py';command+=['--policy',a.policy]
  if a.post_generation_diagnostic:
   assert a.worker=='headline_ours_worker.py';command+=['--post-generation-diagnostic']
  if a.capture_eviction_trace:
   assert a.worker=='headline_ours_worker.py';command+=['--capture-eviction-trace']
  if a.record_main_eviction_trace:
   assert a.worker=='headline_ours_worker.py';command+=['--record-main-eviction-trace']
  state['command']=command
  with (out/'run.log').open('w') as log,(out/'resources.jsonl').open('w') as resources:
   proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['pid']=proc.pid;c.write(out/'status.json',state)
   while proc.poll() is None:
    if time.time()-state['started']>a.timeout:raise TimeoutError('configured job bound')
    if (ROOT/'STOP').exists() or (out/'STOP').exists():raise RuntimeError('owner STOP')
    if c.host_available()<96*2**30:raise RuntimeError('host available below96 GiB')
    phase=json.loads((out/'phase.json').read_text()) if (out/'phase.json').exists() else {'phase':'loading'}
    gpu_rows=c.gpu_state()
    if min_gpu_free_mib:
     free_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
     free_mib={int(g.strip()):int(float(f.strip())) for g,f in (line.split(',') for line in free_text.splitlines())}
     for row in gpu_rows:
      row['free_mib']=free_mib[row['gpu']]
     if any(row['free_mib']<min_gpu_free_mib for row in gpu_rows):
      raise RuntimeError(f'owner GPU free memory below {min_gpu_free_mib} MiB: {gpu_rows}')
    resources.write(json.dumps(dict(unix=time.time(),phase=phase,gpus=gpu_rows,host_available=c.host_available()))+'\n');resources.flush();time.sleep(1)
   state['worker_returncode']=proc.returncode
   assert proc.returncode==0,f'worker exited {proc.returncode}; see run.log'
  result=json.loads((out/'result.json').read_text());assert result['status']=='PASS';state.update(status='PASS',result=result)
 except BaseException as e:
  state.update(status='FAIL',error=repr(e))
  if proc is not None and proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  raise
 finally:
  occupants,observed=wait_for_gpu_release()
  state['final_release_observations']=observed;state['remaining_gpu_occupants']=occupants;state['finished']=time.time()
  idle_rows=json.loads((c.LOAD/'processes.json').read_text()) if (c.LOAD/'processes.json').exists() else []
  already_idle={row['gpu'] for row in idle_rows if c.owned_idle(row['pid'])}
  state['restored']=c.restore_target_idle([gpu for gpu in c.GPUS if gpu not in already_idle])
  c.write(out/'status.json',state)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--workloads');p.add_argument('--label',required=True);p.add_argument('--system',required=True);p.add_argument('--worker',required=True);p.add_argument('--cell',required=True)
 p.add_argument('--nccl-p2p-disable',action='store_true',help='Opt-in P2P-disabled transport after the default NCCL environment reset')
 p.add_argument('--python',default=c.PYTHON);p.add_argument('--ranks',type=int,default=4);p.add_argument('--timeout',type=int,default=3600)
 p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=range(1,6))
 p.add_argument('--llama-threads',type=int,choices=(16,32,64))
 p.add_argument('--llama-cuda-graphs',choices=('on','off'))
 p.add_argument('--llama-graph-reuse',choices=('on','off'))
 p.add_argument('--llama-expert-placement',choices=('legacy_tail','balanced3','balanced4','balanced8','balanced12'))
 p.add_argument('--prefill-optimized',action='store_true');p.add_argument('--prefill-diagnostic',action='store_true');p.add_argument('--prefill-layout-fast',action='store_true');p.add_argument('--post-prefill-diagnostic',action='store_true')
 p.add_argument('--capture-eviction-trace',action='store_true')
 p.add_argument('--policy',choices=('BR','CA','CA_NATIVE','LA_CA_NEAR'))
 p.add_argument('--expert-executor',choices=('h0','native'))
 p.add_argument('--native-prefill',action='store_true')
 p.add_argument('--prefetch-off',action='store_true')
 p.add_argument('--h2d-serial-ablation',action='store_true')
 p.add_argument('--ours-final',action='store_true')
 p.add_argument('--post-generation-diagnostic',action='store_true')
 p.add_argument('--record-main-eviction-trace',action='store_true')
 p.add_argument('--decode-layout-fast',action='store_true')
 p.add_argument('--legacy-decode-layout',action='store_true')
 main(p.parse_args())
