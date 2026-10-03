"""Untimed transport validation for the physical E2E packet."""
import os,json,argparse
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
boot=pin_rank_before_cuda_import()
import torch
import torch.distributed as dist
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(1);torch.cuda.set_device(0)
dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
r=dist.get_rank();w=dist.get_world_size()
x=torch.cat([torch.full((1024,),r*100+d,device='cuda',dtype=torch.float32) for d in range(w)])
y=torch.empty_like(x);dist.all_to_all_single(y,x);torch.cuda.synchronize()
assert all(torch.equal(y[s*1024:(s+1)*1024],torch.full((1024,),s*100+r,device='cuda')) for s in range(w))
(a.output/f'rank{r}.json').write_text(json.dumps(dict(status='PASS',rank=r,world=w,boot=boot,payload_valid=True,nccl_env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}),indent=2)+'\n')
dist.barrier();dist.destroy_process_group()
