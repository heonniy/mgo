"""Eight-rank staging worker overlap with NCCL/GEMM, with content validation."""
from env_offload_worker import *
from mgo_v2.pinned_h2d import PriorityH2DScheduler
from mgo_v2.phase_timing import union_ms,intersection_ms

def main(a):
 rank=int(os.environ['RANK']);torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2)
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 cache=torch.empty((16,EB//2),device='cuda',dtype=torch.bfloat16)
 host=[torch.full((EB//2,),float(rank*16+i),dtype=torch.bfloat16) for i in range(16)]
 x=torch.randn((4096,4096),device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
 send=torch.full((8*256,2048),float(rank),device='cuda',dtype=torch.bfloat16);recv=torch.empty_like(send)
 dist.all_to_all_single(recv,send);torch.mm(x,x,out=y);torch.cuda.synchronize();rows=[]
 for repeat in range(3):
  q=PriorityH2DScheduler(cache,profile=True,autostart=False);dist.barrier()
  for i in range(16):q.enqueue_demand(i,rank*16+i,[host[i]])
  begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
  for _ in range(4):dist.all_to_all_single(recv,send);torch.mm(x,x,out=y)
  q.start()
  for _ in range(128):dist.all_to_all_single(recv,send);torch.mm(x,x,out=y)
  end.record();q.synchronize();torch.cuda.synchronize()
  for i in range(16):assert bool((cache[i]==rank*16+i).all())
  assert bool((recv.view(8,256,2048)[:,0,0]==torch.arange(8,device='cuda')).all())
  copies=[(q.trace_origin.elapsed_time(t.begin),q.trace_origin.elapsed_time(t.done)) for t in q.trace]
  window=[(q.trace_origin.elapsed_time(begin),q.trace_origin.elapsed_time(end))];overlap=intersection_ms(copies,window)
  print(json.dumps(dict(rank=rank,repeat=repeat,copies=q.metrics['copies'],overlap_ms=overlap,window=window,copy_bounds=[min(a for a,b in copies),max(b for a,b in copies)])),flush=True)
  assert q.metrics['copies']==16 and overlap>0
  rows.append(dict(repeat=repeat,H2D_DMA_ms=union_ms(copies),overlap_NCCL_GEMM_window_ms=overlap,content_checks=16))
  q.close()
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,rows=rows));dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args())
