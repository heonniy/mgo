#!/usr/bin/env python3
"""E1: captured variable-size activation traffic only; no model/controller."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import();os.environ.pop('NCCL_P2P_DISABLE',None)
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trace_comm_input import validate_events


def pattern(rows,event,phase,src,dst,device):
    # Integers <=250 are represented exactly by BF16. Every element checked.
    base=(event*17+phase*71+src*31+dst*47)%251
    return ((torch.arange(rows*2048,device=device,dtype=torch.int32)+base)%251).to(torch.bfloat16).view(rows,2048)


def main(a):
    assert BOOT['visible_gpu']==str([0,1,4,5][BOOT['local_rank']])
    want={'NCCL_P2P_LEVEL':'LOC','NCCL_IB_DISABLE':'1'} if a.mode=='R3' else {}
    if getattr(a,'ipc_rebase',False):want['NCCL_CUMEM_ENABLE']='0'
    actual={k:v for k,v in os.environ.items() if k.startswith('NCCL_')}
    assert actual==want,(actual,want)
    source=json.loads(a.counts.read_text());validate_events(source['events'])
    assert source['hidden_size']==2048 and source['row_bytes']==4096
    torch.set_num_threads(1);torch.cuda.set_device(0)
    dist.init_process_group('nccl',device_id=torch.device('cuda:0'));rank=dist.get_rank();assert dist.get_world_size()==4
    packets=[]
    for e in source['events']:
        pair=[]
        for phase_index,phase in enumerate(('dispatch','combine')):
            sends=e[phase+'_send'][rank];recvs=e[phase+'_recv'][rank]
            send=torch.cat([pattern(n,e['source_event'],phase_index,rank,dst,'cuda') for dst,n in enumerate(sends)],dim=0)
            expected=torch.cat([pattern(n,e['source_event'],phase_index,src,rank,'cuda') for src,n in enumerate(recvs)],dim=0)
            recv=torch.empty_like(expected)
            assert send.shape==(sum(sends),2048) and recv.shape==(sum(recvs),2048)
            pair.append((send,recv,expected,sends,recvs))
        packets.append(pair)
    def communicate(packet):
        send,recv,_,sends,recvs=packet
        dist.all_to_all_single(recv,send,output_split_sizes=recvs,input_split_sizes=sends)
    def clear():
        for pair in packets:
            for _,recv,_,_,_ in pair:recv.fill_(float('nan'))
    def validate():
        for i,pair in enumerate(packets):
            for phase,(_,recv,expected,_,counts) in enumerate(pair):
                assert recv.shape[0]==sum(counts) and torch.equal(recv,expected),(i,phase,'payload mismatch')
    # Exactly one untimed full-trace warmup, including deterministic payload validation.
    clear();torch.cuda.synchronize();dist.barrier()
    for pair in packets:
        for packet in pair:communicate(packet)
    torch.cuda.synchronize();validate()
    timings=[tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)) for _ in packets]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    repeats=[]
    for iteration in range(3):
        clear();torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
        wall_start=time.perf_counter();begin.record()
        for pair,(start,middle,stop) in zip(packets,timings):
            start.record();communicate(pair[0]);middle.record();communicate(pair[1]);stop.record()
        end.record();end.synchronize();wall_ms=(time.perf_counter()-wall_start)*1000
        # No payload checks, reductions, quantiles or receipt writes in timing.
        validate()
        event_ms=[dict(dispatch=s.elapsed_time(m),combine=m.elapsed_time(t),pair=s.elapsed_time(t)) for s,m,t in timings]
        repeats.append(dict(iteration=iteration,full_trace_cuda_interval_ms=begin.elapsed_time(end),wall_ms=wall_ms,
                            cumulative_pair_ms=sum(e['pair'] for e in event_ms),events_ms=event_ms,payload_valid=True))
    receipt=dict(status='PASS',rank=rank,boot=BOOT,mode=a.mode,pass_index=a.pass_index,transport_env=actual,
                 ipc_rebase=getattr(a,'ipc_rebase',False),
                 counts_sha256=hashlib.sha256(a.counts.read_bytes()).hexdigest(),events=384,
                 warmup_full_traces=1,timed_full_traces=3,payload_valid=True,
                 no_model=True,no_expert_h2d=True,no_cache_controller=True,nccl_info_in_timing=False,
                 tensor_allocations_outside_timing=True,validation_outside_timing=True,
                 peer_bytes_per_trace={phase:sum((sum(e[phase+'_send'][rank])-e[phase+'_send'][rank][rank])*4096 for e in source['events']) for phase in ('dispatch','combine')},
                 repeats=repeats,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 versions=dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=dist.is_nccl_available()))
    (a.output/f'rank{rank}.json').write_text(json.dumps(receipt,separators=(',',':'))+'\n')
    print(json.dumps(dict(status='PASS',mode=a.mode,pass_index=a.pass_index,rank=rank,cumulative_ms=[r['cumulative_pair_ms'] for r in repeats])),flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counts',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['T0','R3'],required=True);p.add_argument('--pass-index',type=int,choices=[0,1],required=True)
    p.add_argument('--ipc-rebase',action='store_true')
    a=p.parse_args()
    try:main(a)
    except BaseException as exc:
        (a.output/f'failure-rank{BOOT["local_rank"]}.json').write_text(json.dumps(dict(status='FAIL',rank=BOOT['local_rank'],error=repr(exc)),indent=2)+'\n')
        raise
