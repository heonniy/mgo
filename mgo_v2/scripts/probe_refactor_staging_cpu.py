import os,time,ctypes,json,statistics
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.sched_setaffinity(0,{0,1})
import torch
from mgo_v2.pinned_h2d import copy_expert_to_stage
torch.set_num_threads(2)
src=[torch.arange(1572864,dtype=torch.float32).to(torch.bfloat16) for _ in range(3)]
stage=torch.empty(4718592,dtype=torch.bfloat16)
lib=ctypes.CDLL(None);lib.memmove.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.memmove.restype=ctypes.c_void_p
ptrs=[(stage.data_ptr()+i*3145728,t.data_ptr(),t.numel()*t.element_size()) for i,t in enumerate(src)]
def direct():
 for dst,source,size in ptrs:lib.memmove(dst,source,size)
for f in (lambda:copy_expert_to_stage(stage,src),direct):
 f();assert torch.equal(stage,torch.cat(src))
rows={k:[] for k in ('torch','memmove')}
for repeat in range(2):
 for name in (('torch','memmove') if repeat%2==0 else ('memmove','torch')):
  f=(lambda:copy_expert_to_stage(stage,src)) if name=='torch' else direct
  for _ in range(10):f()
  start=time.perf_counter()
  for _ in range(100):f()
  rows[name].append((time.perf_counter()-start)*1000/100)
print(json.dumps({'status':'PASS','scope':'CPU-only synthetic warm pageable 9MiB copy, no model or GPU; descriptive host API probe, not end-to-end timing','repeats_ms_per_copy':rows,'median_ms':{k:statistics.median(v) for k,v in rows.items()}},indent=2))
