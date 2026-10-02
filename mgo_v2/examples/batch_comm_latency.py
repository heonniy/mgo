import os,argparse,json
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
boot=pin_rank_before_cuda_import();os.environ.pop('NCCL_P2P_DISABLE',None)
import torch
import torch.distributed as dist
SIZES=[16384,32768,65536,131072,262144]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--mode',choices=['T0','R3'],required=True);a=p.parse_args()
 assert boot['visible_gpu']==str([0,1,4,5][boot['local_rank']])
 expected={'NCCL_P2P_LEVEL':'LOC','NCCL_IB_DISABLE':'1'} if a.mode=='R3' else {}
 expected['NCCL_CUMEM_ENABLE']='0'
 actual={k:v for k,v in os.environ.items() if k.startswith('NCCL_') and not k.startswith('NCCL_DEBUG')};assert actual==expected,(actual,expected)
 if not a.smoke:assert not any(k.startswith('NCCL_DEBUG') for k in os.environ)
 torch.set_num_threads(1);torch.cuda.set_device(0);dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();assert dist.get_world_size()==4
 results=[]
 for size in (SIZES[:1] if a.smoke else SIZES):
  n=size//2;send=torch.cat([((torch.arange(n,device='cuda',dtype=torch.int32)+rank*31+dst*47)%251).to(torch.bfloat16) for dst in range(4)]);recv=torch.empty_like(send)
  expected_payload=torch.cat([((torch.arange(n,device='cuda',dtype=torch.int32)+src*31+rank*47)%251).to(torch.bfloat16) for src in range(4)])
  samples=[]
  for i in range(1 if a.smoke else 40):
   recv.fill_(float('nan'));dist.barrier();torch.cuda.synchronize()
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   start.record();dist.all_to_all_single(recv,send,async_op=True).wait();end.record();end.synchronize()
   elapsed=start.elapsed_time(end)
   assert torch.equal(recv,expected_payload)
   if not a.smoke and i>=10:samples.append(elapsed)
  results.append(dict(per_peer_payload_bytes=size,peer_tx_bytes_per_rank=3*size,local_self_bytes=size,cuda_interval_ms=samples,payload_valid=True))
  del send,recv
 out=Path(a.output);out.mkdir(exist_ok=True)
 (out/f'rank{rank}.json').write_text(json.dumps(dict(status='PASS',rank=rank,mode=a.mode,boot=boot,smoke=a.smoke,transport_env=actual,results=results),indent=2)+'\n')
 print(json.dumps(dict(status='PASS',rank=rank,mode=a.mode,sizes=len(results))),flush=True)
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
