"""Four authorized B2 prefix diagnostics; preserve owner-disabled GPU workers."""
import os,json,subprocess,time,hashlib
from pathlib import Path
import numpy as np
from prepare_critical_microbench import P,PACKET,ROOT,OLD,write,sha
from mgo_v2.controller import DecodePrefetchController
from mgo_v2.predictor import TransitionPredictor
from batch_comm_common import start_idle_load
import run_timing_stability as h
B2=ROOT/'b2'
NSYS='/home/hwlee/mgo-tools/nsight-2025.3/extracted/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64/nsys'
def prepare():
 B2.mkdir(exist_ok=True);sources={}
 for cache in ('C30','C60'):
  src=OLD/cache/'inputs_B128_H64';dst=B2/cache/'inputs_B128_H8';dst.mkdir(parents=True,exist_ok=True)
  for name in ('selected.npy','weights.npy','offsets.npy','prefill_origins.npy','decode_origins.npy','gates.npy','teacher.npy','requests.json','input_receipt.json','BR_fetches.json','FCA_fetches.json'):
   if not (dst/name).exists():(dst/name).symlink_to(src/name)
   sources[str(src/name)]=sha(src/name)
  receipt=json.loads((src/'receipt.json').read_text());receipt['horizon']=8;receipt['prefix_of']=str(src);write(dst/'receipt.json',receipt)
  a={k:np.load(dst/f'{k}.npy',mmap_mode='r') for k in ('selected','weights','offsets','prefill_origins','decode_origins','gates')};meta=json.loads((src/'input_receipt.json').read_text())
  for policy in ('BR','FCA'):
   target=dst/f'{policy}_P2_proof.json'
   if target.exists():continue
   c=DecodePrefetchController(receipt['capacities'],2,policy,meta['placement_seed'],TransitionPredictor(np.load('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy')));copies=np.zeros(4,np.int64)
   for event in range(9*48):
    lo,hi=a['offsets'][event:event+2];org=a['prefill_origins'] if event<48 else a['decode_origins'];sel=a['selected'][lo:hi]
    out,_,_=c.plan_current(event,sel,a['weights'][lo:hi],org,a['gates'][event])
    for rank,*_ in out[5]:copies[rank]+=1
    if event>=48:
     hist=np.bincount((sel.astype(np.int64)+org.astype(np.int64)[:,None]*128).ravel(),minlength=512).reshape(4,128)
     for rank,*_ in c.plan_prefetch_next(hist):copies[rank]+=1
   c.arena.assert_consistent();assert not c.pending
   digest=lambda x:hashlib.sha256(x.tobytes()).hexdigest()
   write(target,dict(status='PASS',world=4,batch=128,policy=policy,P=2,horizon=8,rank_state_hashes=[digest(c.main.slots[r,:receipt['capacities'][r]]) for r in range(4)],rank_role_hashes=[digest(c.arena.main_physical[r]) for r in range(4)],counters=c.counters,max_copy_counts=copies.tolist()))
 manifest=dict(status='PREPARED',plan_commit='0c2e29c22e7369ef79e2a26178618611778d37b1',gpus=[0,1,4,5],prefix_events=[48,431],selection='First8decode steps fixed before inspecting latency; no tail event expansion.',cells=[[c,p] for c in ('C30','C60') for p in ('BR','FCA')],source_sha256=sources,nsys=NSYS,clock='Host CLOCK_MONOTONIC_RAW; align CUPTI/NVTX to this shared monotonic axis using per-rank NVTX/host anchors and report calibration uncertainty. Never compare raw CUDA-event clocks.',instrumentation='NVTX and host timestamps only; no added CUDA timing events or synchronization. Existing untimed scheduler completion events unchanged.',numerics='Warmup without B2 hooks then capture with hooks; require exact frozen token/cache/route/quota parity, no compilation.',resource_policy='Stop only owned load workers; existing memory/temperature/foreign-process guards. GPU2/3/6/7 remain owner-disabled. Sequential GPU captures.',primary_timing=False)
 write(PACKET/'B2_INSTRUMENTATION_MANIFEST.json',manifest)
def main():
 prepare();os.environ['MGO_RESULT_BRANCH']='codex/policy-regime-20261005';os.environ['MGO_NSYS_BINARY']=NSYS
 state=dict(status='RUNNING',completed=[],started_unix=time.time())
 try:
  for cache,policy in [('C30','BR'),('C30','FCA'),('C60','FCA'),('C60','BR')]:
   base=B2/cache;stage='B2_SHARED_RETRY1';label=f'{stage}_V3_OPT_PF_OVERLAP_{policy}_B128_H8';out=base/label
   if (out/'status.json').exists():assert json.loads((out/'status.json').read_text())['status']=='PASS','preserve failed attempt for explicit recovery'
   else:
    case=json.loads((OLD/cache/f'profile_case_{policy}.json').read_text());case.update(horizon=8,b2_instrumentation=True,b2_allow_other_gpu_jobs=True)
    casepath=base/f'case_{policy}.json';write(casepath,case)
    state['active']=cache+'/'+policy;write(B2/'status.json',state)
    cmd=[h.PYTHON,'-u',str(P/'scripts/run_refactor_profile.py'),'--environment','env1','--root',str(base),'--gpus','0','1','4','5','--stage',stage,'--horizon','8','--arm','V3_OPT_PF_OVERLAP','--batch','128','--policy',policy,'--case-override',str(casepath),'--staging-backend','torch','--unique-combine','--async-metadata-inputs','--fixed-staging-team','--cuda-flush-ms','600000','--defer-idle-restore']
    with (base/f'{label}_driver.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
   state['completed'].append(str(out));write(B2/'status.json',state)
  state['status']='CAPTURES_COMPLETE'
 except BaseException as exc:state.update(status='FAIL',error=repr(exc));raise
 finally:
  state['finished_unix']=time.time();write(B2/'status.json',state);write(PACKET/'B2_CAPTURE_STATUS.json',state);write(B2/'resident_models.json',dict(processes=start_idle_load(),unix=time.time()))
if __name__=='__main__':
 import sys
 if '--prepare-only' in sys.argv:prepare()
 else:main()
