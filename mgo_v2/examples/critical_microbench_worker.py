"""Stage-A CUDA-event measurements; no full model and no primary profiler."""
import argparse,ast,hashlib,json,os,time
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import(strict_numa=False)
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open

def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def event():return torch.cuda.Event(enable_timing=True,external=True)
def stats(xs):
 return dict(median_ms=float(np.median(xs)),p10_ms=float(np.quantile(xs,.1)),p90_ms=float(np.quantile(xs,.9)),min_ms=min(xs),max_ms=max(xs),samples_ms=xs)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True);a=ap.parse_args()
 rank=int(os.environ['RANK']);os.sched_setaffinity(0,[0,24,96,120][rank:rank+1]);torch.set_num_threads(1);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.25)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 assert dist.get_world_size()==4
 data=json.loads(a.inputs.read_text());out=dict(rank=rank,boot=BOOT,affinity=sorted(os.sched_getaffinity(0)),torch_version=torch.__version__,status='RUNNING',records=[])
 def save(stage):
  out['stage']=stage;write(a.output/f'rank{rank}.json',out)
 def measure_collective(spec,packet,delay=0.,delayed=0,stage='A1'):
  matrix=np.array(spec['matrix']);matrix=matrix.T if packet==4096 else matrix
  send=(matrix[rank]* (packet//2)).tolist();recv=(matrix[:,rank]*(packet//2)).tolist()
  x=torch.full((sum(send),),rank+1,dtype=torch.bfloat16,device='cuda');y=torch.empty(sum(recv),dtype=torch.bfloat16,device='cuda')
  sync=torch.zeros(1,device='cuda');s=event();arrival=event();e=event()
  # A captured all-reduce provides a GPU-side common rendezvous. Its duration is
  # outside the samples. Local clocks are not treated as cross-device clocks.
  def op():
   dist.all_reduce(sync)
   s.record()
   if delay and rank==delayed:torch.cuda._sleep(int(cycles_per_ms*delay))
   arrival.record()
   dist.all_to_all_single(y,x,recv,send)
   e.record()
  for _ in range(4):op()
  torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):op()
  completion=[];residency=[];achieved=[]
  for i in range(120):
   g.replay();e.synchronize()
   if i>=20:
    completion.append(s.elapsed_time(e));residency.append(arrival.elapsed_time(e));achieved.append(s.elapsed_time(arrival))
  torch.cuda.synchronize()
  cursor=0
  for src,n in enumerate(recv):
   assert bool((y[cursor:cursor+n]==src+1).all()),'payload mismatch'
   cursor+=n
  out['records'].append(dict(stage=stage,id=spec['id'],packet_bytes=packet,delay_ms=delay,delayed_rank=delayed,matrix=matrix.tolist(),completion=stats(completion),residency=stats(residency),achieved_delay=stats(achieved),payload_valid=True))
  save(stage);del g,x,y
 # Sleep calibrated on each physical GPU, measured achieved delta retained.
 vals=[]
 for _ in range(12):
  s,e=event(),event();s.record();torch.cuda._sleep(1000000);e.record();e.synchronize();vals.append(s.elapsed_time(e))
 cycles_per_ms=1000000/float(np.median(vals[2:]));out['sleep_calibration']=dict(cycles_per_ms=cycles_per_ms,samples_ms=vals)
 for i,spec in enumerate(data['shapes']):
  for packet in ((4120,4096) if i%2==0 else (4096,4120)):measure_collective(spec,packet)
 for spec in data['arrival_matrices']:
  for delayed in range(4):
   for delta in ([0,.1,.25,.5,1,2] if delayed%2==0 else [2,1,.5,.25,.1,0]):measure_collective(spec,4096,delta,delayed,'A2')
 save('A3_PREPARE')
 # Compile the exact runtime function body without importing its model loader
 # (which would repeat rank bootstrap or instantiate unrelated model state).
 source=Path(__file__).resolve().parent/'env_offload_worker.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='expert_kernel')
 module=ast.Module(body=[node],type_ignores=[]);ns={'F':F};exec(compile(module,str(source),'exec'),ns)
 kernel=torch.compile(ns['expert_kernel'],dynamic=True,fullgraph=True)
 out['expert_kernel']=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ast=ast.unparse(node),compile={'dynamic':True,'fullgraph':True})
 model=Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507');index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map'];weights=[]
 for expert in range(8):
  ws=[]
  for proj in ('gate_proj','up_proj','down_proj'):
   name=f'model.layers.0.mlp.experts.{expert}.{proj}.weight'
   with safe_open(str(model/index[name]),framework='pt',device='cpu') as f:ws.append(f.get_tensor(name).to(device='cuda',dtype=torch.bfloat16))
  weights.append(ws)
 out['weight_identity']=dict(model=str(model),layer=0,experts=list(range(8)),index_sha256=hashlib.sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),tensor_sha256=[[hashlib.sha256(w.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for w in ws] for ws in weights])
 torch.manual_seed(20261006+rank)
 # Cover every bundle row outside timing as well as the requested power ladder.
 allrows=sorted(set(data['tau_rows']+[n for b in data['bundles'] for n in b['rows']]))
 xs={n:torch.randn(n,2048,device='cuda',dtype=torch.bfloat16)*.02 for n in allrows}
 for n in allrows:
  for _ in range(3):kernel(xs[n],*weights[0])
 torch.cuda.synchronize()
 from torch._dynamo.utils import counters
 compile_stats=dict(counters['stats']);out['compile_stats']=compile_stats
 def compute(rows):
  def op():
   for i,n in enumerate(rows):kernel(xs[n],*weights[i])
  for _ in range(5):op()
  torch.cuda.synchronize()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):op()
  samples=[]
  for i in range(120):
   s,e=event(),event();s.record();g.replay();e.record();e.synchronize()
   if i>=20:samples.append(s.elapsed_time(e))
  del g
  return stats(samples)
 with torch.inference_mode(),torch._dynamo.config.patch(error_on_recompile=True):
  # inference mode must match compile warmup; set outside main at entry below.
  for n in allrows:
   blocks=[compute([n]),compute([n])]
   drift=abs(blocks[0]['median_ms']-blocks[1]['median_ms'])/np.mean([b['median_ms'] for b in blocks])
   if drift>.02:blocks.append(compute([n]))
   final_drift=abs(blocks[-1]['median_ms']-blocks[-2]['median_ms'])/np.mean([b['median_ms'] for b in blocks[-2:]])
   out['records'].append(dict(stage='A3_TAU',rows=n,primary_ladder=n in data['tau_rows'],blocks=blocks,initial_drift=float(drift),final_drift=float(final_drift),stable=bool(final_drift<=.02)));save('A3_TAU')
  for b in data['bundles']:
   out['records'].append(dict(stage='A3_BUNDLE',**b,timing=compute(b['rows'])));save('A3_BUNDLE')
 assert dict(counters['stats'])==compile_stats,'compilation during timing'
 out.update(status='PASS',peak_gpu_bytes=torch.cuda.max_memory_allocated(),no_compile_in_measure=True);save('COMPLETE');dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 with torch.inference_mode():main()
