"""Physical H2D matrix; NCCL is initialized only after isolated samples finish."""
import argparse
import csv
import datetime
import itertools
import json
import os
from pathlib import Path
import random
import shutil
import socket
import subprocess
import sys
import time

PKG = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PKG), str(PKG/'scripts')]
from pcie_host import GPUS, affinity, write, topology
from pcie_gpu_occupancy import gpu_experiment

EB = 9437184


def matrix():
    cells = []
    def add(name, counts, streams=1, remote=False, overlap=False):
        cells.append(dict(name=name, counts=list(counts), streams=streams, remote=remote, overlap=overlap))
    for rank in range(4):
        q = [0]*4; q[rank] = 2; add(f'two_single_{rank}', q)
    for a,b in itertools.combinations(range(4), 2):
        q = [0]*4; q[a]=q[b]=1; add(f'two_pair_{a}_{b}', q)
    for i,q in enumerate(([2,2,1,1],[1,2,1,2],[2,1,2,1],[1,1,2,2])):
        add(f'six_{i}', q)
    for m in (2,4,6,8,10,14,50,62):
        rank = [m//4+int(r<m%4) for r in range(4)]
        group = []
        for g in range(2):
            total=m//2+int(m%2 and g==0)
            group.extend([total//2,total//2+total%2])
        add(f'scale_{m}_rank',rank);add(f'scale_{m}_group',group)
    for streams in (2,4):
        for name,q in (('rank',[4,4,3,3]),('group',[3,4,3,4])):
            add(f'concurrency_{streams}_{name}',q,streams=streams)
    for i,q in enumerate(([2,2,1,1],[1,2,1,2],[1,1,0,0],[1,0,1,0])):
        add(f'remote_{i}',q,remote=True)
    for i,q in enumerate(([2,2,1,1],[1,2,1,2],[2,1,2,1])):
        add(f'overlap_{i}',q,overlap=True)
    return cells


def worker(a):
    rank=a.rank;os.environ['CUDA_VISIBLE_DEVICES']=str(GPUS[rank])
    os.sched_setaffinity(0,affinity(rank))
    import torch
    import torch.distributed as dist
    from mgo_v2.numa_shared_source import SharedPinnedMapping
    torch.set_num_threads(1);torch.cuda.set_device(0)
    dest=torch.empty((16,EB//2),device='cuda',dtype=torch.bfloat16)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{a.port}',rank=rank,world_size=4,timeout=datetime.timedelta(seconds=600))
    node=rank//2;root=Path(a.pool);root.mkdir(parents=True,exist_ok=True)
    mappings={}
    if rank%2==0:
        pool=SharedPinnedMapping(root/f'node{node}.bin',16*EB,node,create=True)
        pool.tensor.fill_(node+1);mappings[node]=pool
    dist.barrier()
    receipts=[]
    for source_node in (0,1):
        if source_node not in mappings:
            mappings[source_node]=SharedPinnedMapping(root/f'node{source_node}.bin',16*EB,source_node)
        receipts.append(mappings[source_node].register())
    write(a.out/f'host_rank{rank}.json',dict(rank=rank,gpu=GPUS[rank],affinity=sorted(os.sched_getaffinity(0)),sources=receipts))
    rows={n:p.tensor.view(16,EB//2) for n,p in mappings.items()}
    streams=[torch.cuda.Stream() for _ in range(4)]
    starts=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ends=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
    records=[]
    def release():
        torch.cuda.synchronize();dist.barrier()
        stamp=torch.tensor([time.perf_counter_ns()+10_000_000 if rank==0 else 0],dtype=torch.int64)
        dist.broadcast(stamp,0);start=int(stamp.item())
        while time.perf_counter_ns()<start:pass
        return start
    def run(cell,repeat,nccl=None,send=None,recv=None):
        count=cell['counts'][rank];concurrency=cell['streams'];source_node=1-node if cell['remote'] else node
        start=release();comm=None
        if cell['overlap']:
            comm=dist.all_to_all_single(recv,send,group=nccl,async_op=True)
        for s in range(concurrency):
            with torch.cuda.stream(streams[s]):starts[s].record()
        for index in range(count):
            with torch.cuda.stream(streams[index%concurrency]):dest[index].copy_(rows[source_node][index],non_blocking=True)
        for s in range(concurrency):
            with torch.cuda.stream(streams[s]):ends[s].record()
        for s in range(concurrency):ends[s].synchronize()
        end=time.perf_counter_ns()
        service=max(starts[s].elapsed_time(ends[s]) for s in range(concurrency)) if count else 0
        if comm is not None:comm.wait();torch.cuda.synchronize()
        if repeat<0 and count:
            assert torch.equal(dest[:count],torch.full_like(dest[:count],source_node+1))
        records.append(dict(cell=cell['name'],repeat=repeat,rank=rank,gpu=GPUS[rank],counts=cell['counts'],
                            streams=concurrency,remote=cell['remote'],overlap=cell['overlap'],
                            start_ns=start,ready_ns=end,bytes=count*EB,service_ms=service,
                            nccl_initialized=nccl is not None))
    cells=matrix();rng=random.Random(20261009)
    isolated=[c for c in cells if not c['overlap']]
    for repeat in range(-5,30):
        order=isolated.copy();rng.shuffle(order)
        for cell in order:run(cell,repeat)
    write(a.out/f'isolated_rank{rank}.json',records)
    # Separate calibration/overlap sensitivity: no NCCL object existed above.
    nccl=dist.new_group(backend='nccl',timeout=datetime.timedelta(seconds=600))
    pair_records=[]
    for src,dst in itertools.permutations(range(4),2):
        pair_group=dist.new_group(ranks=sorted([src,dst]),backend='nccl')
        for size in (4096,1048576,9437184):
            payload=torch.full((size,),rank+1,device='cuda',dtype=torch.uint8)
            received=torch.empty_like(payload)
            for repeat in range(-5,30):
                start=release()
                if rank in (src,dst):
                    # Full collective key avoids torch 2.5's P2P shutdown key
                    # being interpreted as a CUDA ordinal under single-GPU CVD.
                    dist.broadcast(payload,src=src,group=pair_group)
                torch.cuda.synchronize();end=time.perf_counter_ns()
                if rank==dst:
                    if repeat<0:assert bool((payload==src+1).all())
                    pair_records.append(dict(src=src,dst=dst,bytes=size,repeat=repeat,start_ns=start,ready_ns=end))
        dist.barrier()
        if rank in (src,dst):dist.destroy_process_group(pair_group)
        dist.barrier()
    write(a.out/f'pairs_rank{rank}.json',pair_records)
    send=torch.full((16*1024*1024,),rank+1,device='cuda',dtype=torch.uint8);recv=torch.empty_like(send)
    sensitivity=[c for c in cells if c['overlap']]
    for repeat in range(-5,30):
        order=sensitivity.copy();rng.shuffle(order)
        for cell in order:run(cell,repeat,nccl,send,recv)
    write(a.out/f'all_rank{rank}.json',records)
    dist.barrier()
    for pool in mappings.values():pool.unregister()
    write(a.out/f'teardown_rank{rank}.json',dict(status='PASS',unregistered=True))
    dist.destroy_process_group(nccl)
    dist.barrier()
    dist.destroy_process_group()


def summarize(out):
    import numpy as np
    records=[row for r in range(4) for row in json.loads((out/f'all_rank{r}.json').read_text())]
    fields=list(records[0])
    with (out/'microbench_raw.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    cells=[];rng=np.random.default_rng(20261009)
    for cell in matrix():
        timed=[r for r in records if r['cell']==cell['name'] and r['repeat']>=0]
        assert len(timed)==120
        if not cell['overlap']:assert not any(r['nccl_initialized'] for r in timed)
        makespan=[];groups=[[],[]]
        for repeat in range(30):
            rr=sorted([r for r in timed if r['repeat']==repeat],key=lambda r:r['rank'])
            assert len(set(r['start_ns'] for r in rr))==1
            start=rr[0]['start_ns'];ready=max(r['ready_ns'] for r in rr if r['bytes'])
            makespan.append((ready-start)/1e6)
            for g in range(2):
                ranks=[r for r in rr[g*2:g*2+2] if r['bytes']]
                groups[g].append(sum(r['bytes'] for r in ranks)/(max(r['ready_ns'] for r in ranks)-start) if ranks else 0)
        def stats(v):
            v=np.array(v);return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),
                                      median=float(np.median(v)),p10=float(np.percentile(v,10)),p90=float(np.percentile(v,90)),p99=float(np.percentile(v,99)))
        boot=np.median(rng.choice(makespan,(10000,30),replace=True),axis=1)
        cells.append(dict(**cell,n=30,makespan_ms=stats(makespan),median_ci95_ms=np.percentile(boot,[2.5,97.5]).tolist(),
                          group_GBps=[stats(x) for x in groups],
                          ranks=[dict(gpu=GPUS[r],service_ms=stats([x['service_ms'] for x in timed if x['rank']==r]),
                                      completion_ms=stats([(x['ready_ns']-x['start_ns'])/1e6 for x in timed if x['rank']==r])) for r in range(4)]))
    pairs=[row for r in range(4) for row in json.loads((out/f'pairs_rank{r}.json').read_text()) if row['repeat']>=0]
    with (out/'g2g_pair_raw.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    cost=np.zeros((4,4),float);pair_summary=[]
    for src,dst in itertools.permutations(range(4),2):
        means=[]
        for size in (4096,1048576,9437184):
            values=[r['ready_ns']-r['start_ns'] for r in pairs if r['src']==src and r['dst']==dst and r['bytes']==size]
            means.append(float(np.median(values)))
        slope=max((means[-1]-means[0])/(9437184-4096),1e-6);cost[src,dst]=slope
        pair_summary.append(dict(src=src,dst=dst,bytes=[4096,1048576,9437184],median_ns=means,ns_per_byte=slope))
    costs=np.rint(cost/cost[cost>0].min()*1000).astype(np.int64)
    write(out/'peer_costs.json',dict(status='PASS',matrix=costs.tolist(),method='measured directional 9MiB-minus-4KiB transfer slope, integer scale 1000',pairs=pair_summary))
    result=dict(status='PASS',payload_bytes=EB,gpus=list(GPUS),warmups=5,timed_samples=30,seed=20261009,
                source='one shared pinned pool per NUMA node, both registered by each rank for local/remote controls',
                isolated_nccl_never_initialized=True,cells=cells)
    write(out/'microbench_summary.json',result)
    write(out/'result.json',dict(status='PASS',cells=len(cells),timed_rank_samples=30*4*len(cells),pair_calibration_samples=len(pairs)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int);p.add_argument('--pool');p.add_argument('--port',type=int)
    a=p.parse_args()
    if a.rank is not None:worker(a);return
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'topology_before.json',topology())
    pool=Path('/dev/shm')/f'esjung_pcie_micro_{os.getpid()}'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[];logs=[]
    try:
        with gpu_experiment('PCIe H2D microbenchmark and pair calibration'):
            for rank in range(4):
                log=(a.out/f'worker_rank{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[rank]),OMP_NUM_THREADS='1',NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(a.out/f'nccl_rank{rank}.log'))
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--out',str(a.out),'--rank',str(rank),'--pool',str(pool),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+600
            while any(x.poll() is None for x in processes):
                if any(x.poll() not in (None,0) for x in processes):raise RuntimeError('microbenchmark worker failed; inspect rank logs')
                if time.monotonic()>deadline:raise TimeoutError('microbenchmark exceeded 600 seconds')
                time.sleep(.2)
            if any(x.returncode for x in processes):raise RuntimeError('microbenchmark worker failed')
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()
        shutil.rmtree(pool,ignore_errors=True)
    summarize(a.out);write(a.out/'topology_after.json',topology())


if __name__=='__main__':main()
