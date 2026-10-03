#!/usr/bin/env python3
"""Model-free pair round trip and pinned one-expert H2D calibration."""
import os,argparse,json
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import()
import torch
import torch.distributed as dist
N_ROWS=(1,2,4,8,16,32,64,128,256,512)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--mode',choices=['T0','R3'],required=True);p.add_argument('--kind',choices=['pair','h2d'],required=True);a=p.parse_args()
    expected={'NCCL_CUMEM_ENABLE':'0'}
    if a.mode=='R3':expected.update(NCCL_P2P_LEVEL='LOC',NCCL_IB_DISABLE='1')
    actual={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}
    assert actual==expected,(actual,expected)
    assert BOOT['visible_gpu']==str([0,1,4,5][BOOT['local_rank']])
    torch.set_num_threads(1);torch.cuda.set_device(0)
    dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();assert dist.get_world_size()==4
    records=[]
    for case in (N_ROWS if a.kind=='pair' else ('single-rank','four-rank-concurrent')):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        if a.kind=='pair':
            n=int(case);payload=((torch.arange(n*2048,device='cuda',dtype=torch.int32)%251).to(torch.bfloat16)).reshape(n,2048)
            received=torch.empty_like(payload);returned=torch.empty_like(payload)
        else:
            active=rank==0 or case=='four-rank-concurrent'
            if active:
                host=torch.empty((3,2048,768),dtype=torch.bfloat16,pin_memory=True).fill_(17+rank)
                device=torch.empty_like(host,device='cuda');assert host.numel()*host.element_size()==9437184
        samples=[]
        for i in range(40):
            if a.kind=='pair':received.fill_(float('nan'));returned.fill_(float('nan'))
            elif active:device.fill_(float('nan'))
            torch.cuda.synchronize()
            # GPU stream dependency on the all-rank barrier releases concurrent
            # copies together; no host synchronize between barrier and copy.
            dist.barrier(async_op=True).wait()
            start.record()
            if a.kind=='pair':
                if rank==0:dist.isend(payload,dst=1).wait()
                elif rank==1:dist.irecv(received,src=0).wait()
                if rank==1:dist.isend(received,dst=0).wait()
                elif rank==0:dist.irecv(returned,src=1).wait()
            elif active:device.copy_(host,non_blocking=True)
            end.record();end.synchronize()
            elapsed=start.elapsed_time(end)
            # Payload checks are strictly outside the timed interval.
            if a.kind=='pair':
                if rank==0:assert torch.equal(returned,payload)
                elif rank==1:assert torch.equal(received,payload)
            elif active:assert bool(torch.all(device==17+rank))
            if i>=10:samples.append(elapsed)
        records.append(dict(case=case,cuda_ms=samples,payload_valid=True))
        if a.kind=='pair':del payload,received,returned
        elif active:del host,device
    result=dict(status='PASS',rank=rank,boot=BOOT,mode=a.mode,kind=a.kind,world_size=4,
                transport_env=actual,no_model=True,warmups=10,samples=30,payload_valid=True,
                origin_rank=0,remote_owner_rank=1,physical_gpus=[0,1,4,5],row_bytes=4096,
                expert_bytes=9437184,records=records)
    (Path(a.output)/f'rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
