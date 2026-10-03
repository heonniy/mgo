#!/usr/bin/env python3
"""T0 actual/self-only communication traces: one warmup and five timed runs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
from mgo_v2.bootstrap import pin_rank_before_cuda_import
BOOT=pin_rank_before_cuda_import();os.environ.pop('NCCL_P2P_DISABLE',None)
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from trace_comm_input import validate_events


def pattern(rows,event,phase,src,dst):
    base=(event*17+phase*71+src*31+dst*47)%251
    return ((torch.arange(rows*2048,device='cuda',dtype=torch.int32)+base)%251).to(torch.bfloat16).view(rows,2048)


def check_controls(actual,control):
    validate_events(actual['events'])
    assert actual['rho']==control['rho']
    assert len(control['events'])==384
    for a,b in zip(actual['events'],control['events']):
        assert (a['source_event'],a['layer'],a['step'])==(b['source_event'],b['layer'],b['step'])
        for phase in ('dispatch','combine'):
            for direction in ('send','recv'):
                x=a[phase+'_'+direction];y=b[phase+'_'+direction]
                assert y==[[v if i==j else 0 for j,v in enumerate(row)] for i,row in enumerate(x)]
            assert b[phase+'_recv']==[list(t) for t in zip(*b[phase+'_send'])]


def run_condition(source,rank):
    packets=[]
    for e in source['events']:
        pair=[]
        for phase_index,phase in enumerate(('dispatch','combine')):
            sends=e[phase+'_send'][rank];recvs=e[phase+'_recv'][rank]
            send=torch.cat([pattern(n,e['source_event'],phase_index,rank,dst) for dst,n in enumerate(sends)],dim=0)
            expected=torch.cat([pattern(n,e['source_event'],phase_index,src,rank) for src,n in enumerate(recvs)],dim=0)
            receive=torch.empty_like(expected)
            assert send.shape==(sum(sends),2048) and receive.shape==(sum(recvs),2048)
            pair.append((send,receive,expected,sends,recvs))
        packets.append(pair)
    def communicate(packet):
        send,receive,_,sends,recvs=packet
        dist.all_to_all_single(receive,send,output_split_sizes=recvs,input_split_sizes=sends)
    def clear():
        for pair in packets:
            for _,receive,_,_,_ in pair:receive.fill_(float('nan'))
    def validate():
        for i,pair in enumerate(packets):
            for phase,(_,receive,expected,_,_) in enumerate(pair):assert torch.equal(receive,expected),(i,phase,'payload mismatch')
    clear();torch.cuda.synchronize();dist.barrier()
    for pair in packets:
        for packet in pair:communicate(packet)
    torch.cuda.synchronize();validate()
    timers=[tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)) for _ in packets]
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    repeats=[]
    for iteration in range(5):
        clear();torch.cuda.synchronize();dist.barrier();torch.cuda.synchronize()
        wall_start=time.perf_counter();begin.record()
        for pair,(start,middle,stop) in zip(packets,timers):
            start.record();communicate(pair[0]);middle.record();communicate(pair[1]);stop.record()
        end.record();end.synchronize();wall_ms=(time.perf_counter()-wall_start)*1000
        validate()
        event_ms=[dict(dispatch=a.elapsed_time(b),combine=b.elapsed_time(c),pair=a.elapsed_time(c)) for a,b,c in timers]
        repeats.append(dict(iteration=iteration,whole_cuda_ms=begin.elapsed_time(end),whole_wall_ms=wall_ms,events_ms=event_ms,payload_valid=True))
    return dict(kind=source['kind'],warmup_full_traces=1,timed_full_traces=5,events=384,collective_calls_per_trace=768,payload_valid=True,
                peer_bytes_per_trace=sum((sum(e[phase+'_send'][rank])-e[phase+'_send'][rank][rank])*4096 for e in source['events'] for phase in ('dispatch','combine')),
                self_bytes_per_trace=sum(e[phase+'_send'][rank][rank]*4096 for e in source['events'] for phase in ('dispatch','combine')),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),repeats=repeats)


def main(a):
    assert BOOT['visible_gpu']==str([0,1,4,5][BOOT['local_rank']])
    env={k:v for k,v in os.environ.items() if k.startswith('NCCL_')};assert env=={'NCCL_CUMEM_ENABLE':'0'},env
    actual=json.loads(a.counts.read_text());control=json.loads(a.self_counts.read_text());check_controls(actual,control)
    assert control['actual_sha256']==hashlib.sha256(a.counts.read_bytes()).hexdigest()
    torch.set_num_threads(1);torch.cuda.set_device(0);dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
    rank=dist.get_rank();assert dist.get_world_size()==4
    order=['actual','self'] if a.pass_index==0 else ['self','actual']
    sources={'actual':actual,'self':control};conditions={}
    for kind in order:conditions[kind]=run_condition(sources[kind],rank)
    result=dict(status='PASS',rank=rank,rho=actual['rho'],pass_index=a.pass_index,condition_order=order,boot=BOOT,transport_env=env,
                actual_counts_sha256=hashlib.sha256(a.counts.read_bytes()).hexdigest(),self_counts_sha256=hashlib.sha256(a.self_counts.read_bytes()).hexdigest(),
                no_model=True,no_h2d=True,no_cache_controller=True,validation_outside_timing=True,allocations_outside_timing=True,nccl_info_in_timing=False,conditions=conditions)
    (a.output/f'rank{rank}.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counts',type=Path,required=True);p.add_argument('--self-counts',type=Path,required=True);p.add_argument('--pass-index',type=int,choices=[0,1],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a)
    except BaseException as exc:(a.output/f'failure-rank{BOOT["local_rank"]}.json').write_text(json.dumps(dict(status='FAIL',error=repr(exc)))+'\n');raise
