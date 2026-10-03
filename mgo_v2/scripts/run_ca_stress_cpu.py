"""Adaptive CPU-only search; every scheduled task belongs to the frozen matrix."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[key]='1'
import multiprocessing as mp,queue,time,resource,json,hashlib,statistics
from pathlib import Path
import numpy as np
import psutil
from ca_stress_common import *
from ca_stress_cpu import Pool,proxy_scores,candidate_job,exact

def worker(tasks,results,cpu):
 os.sched_setaffinity(0,{cpu});resource.setrlimit(resource.RLIMIT_AS,(96*2**30,96*2**30));resource.setrlimit(resource.RLIMIT_DATA,(8*2**30,8*2**30));pool=None;orders={}
 while True:
  job=tasks.get()
  if job is None:return
  start=time.monotonic()
  try:
   if pool is None or pool.name!=job['dataset']:pool=Pool(job['dataset'])
   if job['kind']=='A':
    world,batch,seed=job['world'],job['batch'],job['sample_seed'];n=world*batch;dest=ROOT/'stage_a'/f'{pool.name}_R{world}_B{batch}'/f's{seed:03d}.json'
    if dest.exists():
     old=json.loads(dest.read_text());assert old['status']=='PASS' and old['pool_receipt_sha256']==sha(pool.path/'receipt.json') and old['proxy_source_sha256']==sha(PACKAGE/'scripts/ca_stress_cpu.py')
    else:
     if n not in orders:orders[n]=np.array([np.random.default_rng(dp).permutation(n) for dp in range(256)],np.int64)
     sample=np.random.default_rng(seed).permutation(pool.n)[:n]
     values=proxy_scores(pool.a['proxy_routes'],sample,orders[n],world,batch)
     write(dest,dict(status='PASS',scores=values.tolist(),job=job,pool_receipt_sha256=sha(pool.path/'receipt.json'),proxy_source_sha256=sha(PACKAGE/'scripts/ca_stress_cpu.py')))
    result=dict(path=str(dest))
   else:result=candidate_job(pool,job)
   results.put(dict(status='PASS',job=job,result=result,seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
  except BaseException as exc:results.put(dict(status='FAIL',job=job,error=repr(exc)));return

def drive(jobs,stage):
 cpus=sorted(os.sched_getaffinity(0));reserved=min(16,max(1,len(cpus)//8));cpus=cpus[reserved:]
 limit=min(len(cpus),int(max(0,psutil.virtual_memory().available-320*2**30)//(8*2**30)));assert limit>=1
 context=mp.get_context('spawn');tasks=context.Queue();results=context.Queue();workers=[];active=min(8,limit);previous=None;adapt=True;history=[];completed=0;peak=0
 def grow(n):
  while len(workers)<n:
   p=context.Process(target=worker,args=(tasks,results,cpus[len(workers)]));p.start();workers.append(p)
 try:
  for_start=0
  while for_start<len(jobs):
   grow(active);wave=jobs[for_start:for_start+active];start=time.monotonic()
   for job in wave:tasks.put(job)
   received=[]
   while len(received)<len(wave):
    if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
    if psutil.virtual_memory().available<256*2**30:raise RuntimeError('Host memory reserve guard')
    rss=sum(psutil.Process(p.pid).memory_info().rss for p in workers if p.is_alive());peak=max(peak,rss)
    try:item=results.get(timeout=2)
    except queue.Empty:
     if any(not p.is_alive() for p in workers):raise RuntimeError('CPU worker exited unexpectedly')
     continue
    assert item['status']=='PASS',item;received.append(item)
   elapsed=time.monotonic()-start;work=sum(128+j['world']*j['batch'] for j in wave);rate=work/elapsed;completed+=len(wave)
   history.append(dict(concurrency=active,jobs=len(wave),seconds=elapsed,normalized_work_per_second=rate,completed=completed))
   write(ROOT/f'CPU_{stage}_progress.json',dict(status='RUNNING',completed=completed,total=len(jobs),active_limit=active,max_limit=limit,workers=len(workers),reserved_vcpus=reserved,peak_sum_RSS_bytes_double_counts_shared_pages=peak,history=history,updated_unix=time.time()))
   if adapt and len(wave)==active:
    if previous is None or rate>=previous['rate']*1.05:
     previous=dict(rate=rate,limit=active);active=min(limit,active*2)
    else:active=previous['limit'];adapt=False
    if active==limit:adapt=False
   for_start+=len(wave)
   print(stage,completed,'/',len(jobs),'concurrency',active,flush=True)
  write(PACKET/f'CPU_{stage}_resources.json',dict(status='PASS',completed=completed,max_limit=limit,final_concurrency=active,history=history,peak_sum_RSS_bytes_double_counts_shared_pages=peak,guards='8-GiB private-data and 96-GiB address-space limit per worker; host available >=256 GiB; single-thread BLAS/Numba; disjoint worker vCPUs',throughput_note='normalized by 128+R*B to account approximately for route volume; includes useful first-wave work, not extra repetitions'))
  write(ROOT/f'CPU_{stage}_progress.json',json.loads((PACKET/f'CPU_{stage}_resources.json').read_text()))
 except BaseException as exc:
  write(ROOT/f'CPU_{stage}_progress.json',dict(status='FAILED',completed=completed,total=len(jobs),error=repr(exc),history=history));raise
 finally:
  for _ in workers:tasks.put(None)
  for p in workers:
   p.join(timeout=5)
   if p.is_alive():p.terminate();p.join(timeout=5)

def baseline_check():
 pool=Pool('MATH');world=4;batch=8;order=np.array([i for r in range(world) for i in range(r,world*batch,world)],np.int64);arrays=pool.pack(order,world,batch);old=SOURCE/'packed/MATH/R4_B8'
 for key,value in arrays.items():assert np.array_equal(value,np.load(old/(key+'.npy'),mmap_mode='r')),key
 result=[]
 for policy in ['BR','CA']:
  now=exact(arrays,world,30,policy,42);path=SOURCE/'cells'/f'MATH_h256_R4_B8_c30_gate_s0_{policy}_seed42.json';before=json.loads(path.read_text())
  assert now['final_state_sha256']==before['final_state_sha256']
  for key in ['peer_bytes','H2D_bytes','exact_global_hits','exact_local_hits','reload_fetches']:assert now['full'][key]==before['full'][key],(policy,key)
  result.append(dict(policy=policy,source_path=str(path),source_sha256=sha(path),state_hash=now['final_state_sha256'],quota_max_minus_min=now['quota_max_minus_min']))
 write(PACKET/'exact_replay_validation.json',dict(status='PASS',packed_arrays_identical=True,original512_not_recaptured=True,substitution=False,eviction='Gate W128',rows=result))

def main():
 baseline_check()
 jobs=[dict(kind='A',dataset=dataset,world=world,batch=batch,sample_seed=seed) for dataset in ['MATH','ShareGPT'] for world in [4,8] for batch in [8,16,32,64] for seed in range(256)];jobs.sort(key=lambda j:-(j['world']*j['batch']))
 drive(jobs,'A');retained=[]
 for dataset in ['MATH','ShareGPT']:
  for world in [4,8]:
   for batch in [8,16,32,64]:
    values=np.array([json.loads((ROOT/'stage_a'/f'{dataset}_R{world}_B{batch}'/f's{seed:03d}.json').read_text())['scores'] for seed in range(256)])
    assert values.shape==(256,256) and np.isfinite(values).all()
    order=np.argsort(-values.ravel(),kind='stable')[:32]
    for index in order:
     sample,dp=divmod(int(index),256);retained.append(dict(kind='B',dataset=dataset,world=world,batch=batch,sample_seed=sample,dp_seed=dp,proxy_score=float(values[sample,dp])))
 assert len(retained)==512;write(ROOT/'retained.json',retained);write(PACKET/'retained_candidates.json',retained)
 retained.sort(key=lambda j:-(j['world']*j['batch']));drive(retained,'B')
if __name__=='__main__':main()
