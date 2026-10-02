#!/usr/bin/env python3
"""One 32-KiB P2P smoke; cuMem off is diagnostic only."""
import argparse
import json
import os
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import();os.environ.pop('NCCL_P2P_DISABLE',None)
import torch
import torch.distributed as dist


def main(a):
    assert BOOT['visible_gpu']==str([0,1,4,5][BOOT['local_rank']])
    actual={k:v for k,v in os.environ.items() if k.startswith('NCCL_') and not k.startswith('NCCL_DEBUG')}
    assert actual==({'NCCL_CUMEM_ENABLE':'0'} if a.cumem_disabled else {}),actual
    torch.set_num_threads(1);torch.cuda.set_device(0)
    dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
    rank=dist.get_rank();assert dist.get_world_size()==4
    n=32768//2
    send=torch.full((4*n,),float(rank+1),device='cuda',dtype=torch.bfloat16);recv=torch.empty_like(send)
    dist.barrier();torch.cuda.synchronize()
    dist.all_to_all_single(recv,send,async_op=True).wait();torch.cuda.synchronize()
    for src in range(4):assert torch.all(recv[src*n:(src+1)*n]==src+1)
    data=dict(status='PASS',rank=rank,boot=BOOT,cumem_disabled=a.cumem_disabled,transport_env=actual,
              per_peer_payload_bytes=32768,payload_valid=True,
              versions=dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version())))
    (a.output/f'rank{rank}.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',rank=rank)),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cumem-disabled',action='store_true');a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        (a.output/f'failure-rank{BOOT["local_rank"]}.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)),indent=2)+'\n')
        raise
