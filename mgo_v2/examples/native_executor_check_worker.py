"""Bounded numerical, stream-dependency and host-loop migration checks."""
from env_offload_worker import *
from mgo_v2.native_expert import NativeExpertExecutor, load_native_expert
from mgo_v2.pinned_h2d import PriorityH2DScheduler
from types import SimpleNamespace


def main(a):
 rank=int(os.environ['RANK']);torch.cuda.set_device(0);torch.set_num_threads(2)
 torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(412)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 native=load_native_expert();compiled=torch.compile(expert_kernel,dynamic=True,fullgraph=True)
 cache=torch.randn((32,4718592),device='cuda',dtype=torch.bfloat16)*.02
 errors=[];timings=[]
 def reference(inp,rw,slots,rows,cols):
  result=[]
  for s,r,c in zip(slots,rows,cols):
   w=cache[s];result.append(compiled(inp[r],w[:1572864].view(768,2048),w[1572864:3145728].view(768,2048),w[3145728:].view(2048,768))*rw[r,c,None])
  return result
 for n in (1,3,5,17,65):
  inp=torch.randn((n*32,2048),device='cuda',dtype=torch.bfloat16)
  rw=torch.rand((n*32,8),device='cuda',dtype=torch.bfloat16)
  slots=list(reversed(range(32)))
  rows=[torch.arange(i*n,(i+1)*n,device='cuda') for i in range(32)]
  cols=[torch.full((n,),i%8,device='cuda',dtype=torch.int64) for i in range(32)]
  expected=reference(inp,rw,slots,rows,cols)
  actual=native.execute_wave(cache,inp,rw,slots,rows,cols)
  ref=torch.cat(expected).float();out=torch.cat(actual).float();diff=out-ref
  rel=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(ref).clamp_min(1e-12))
  err=dict(rows_per_expert=n,relative_l2=rel,max_abs=float(diff.abs().max()),finite=bool(torch.isfinite(out).all()))
  assert err['finite'] and rel<.01,err
  errors.append(err)
  assert native.execute_wave(cache,inp,rw,[],[],[])==[]
  if n not in (3,5,17):continue
  for name in ('h0','cpp_single','cpp_wave'):
   def call():
    if name=='h0':return reference(inp,rw,slots,rows,cols)
    if name=='cpp_single':return [native.execute_wave(cache,inp,rw,[s],[r],[c])[0] for s,r,c in zip(slots,rows,cols)]
    return native.execute_wave(cache,inp,rw,slots,rows,cols)
   for _ in range(3):call()
   for repeat in range(2):
    dist.barrier();torch.cuda.synchronize();start=time.perf_counter();cpu=time.thread_time()
    for _ in range(20):call()
    cpu=time.thread_time()-cpu;torch.cuda.synchronize();wall=time.perf_counter()-start
    timings.append(dict(rows_per_expert=n,backend=name,repeat=repeat,experts=32,iterations=20,wall_us_per_expert=wall/640*1e6,cpu_us_per_expert=cpu/640*1e6))
 # Real pinned H2D, pending path, non-default compute stream and immediate
 # overwrite after the wave: the scheduler must honor shared slot-use events.
 small=cache[:3].clone();scheduler=PriorityH2DScheduler(small,direct_pinned=True)
 source=[small[i].cpu().pin_memory() for i in range(3)]
 keys=np.array([0,1,2]);rt=SimpleNamespace(cache=small,h2d=scheduler,keys=keys,ready_metrics=dict(waits=0,ready_before_first_wait=0))
 groups=[(i,rows[i],cols[i],i) for i in range(3)]
 executor=NativeExpertExecutor(max_experts=2)
 # Force the valid not-ready snapshot branch once; real wait_for_slot still
 # supplies the dependency, and any accidental host-DMA wait is rejected.
 original_ready=scheduler.ready_many;calls=[0]
 def ready(slots):
  calls[0]+=1
  return [False]*len(slots) if calls[0]==1 else original_ready(slots)
 scheduler.ready_many=ready
 scheduler.wait_slots=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('unexpected host wait'))
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 for i in range(3):scheduler.enqueue_demand(i,i,(source[i],))
 with torch.cuda.stream(stream):
  output=executor.compute(rt,('current',inp,None,rw),dict(groups=groups),0)
  baseline=native.execute_wave(torch.stack(source).to('cuda'),inp,rw,[0,1,2],rows[:3],cols[:3])
  # Independent source copy on this stream does not read overwritten cache.
  for i in range(3):scheduler.enqueue_demand(i,128+i,(torch.zeros_like(source[i],pin_memory=True),))
 stream.synchronize();scheduler.synchronize()
 for x,y in zip(output,baseline):torch.testing.assert_close(x,y,rtol=0,atol=0)
 assert bool((small==0).all()) and executor.waits>=1
 scheduler.close()
 row=dict(status='PASS',rank=rank,errors=errors,timings=timings,not_ready_and_overwrite=True,scope='Synthetic resident experts; not full-model TPOT')
 (a.output/f'rank{rank}.json').write_text(json.dumps(row,indent=2)+'\n');dist.barrier()
 if rank==0:(a.output/'result.json').write_text(json.dumps(dict(status='PASS',ranks=4),indent=2)+'\n')
 dist.destroy_process_group()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell');p.add_argument('--output',type=Path,required=True);main(p.parse_args())
