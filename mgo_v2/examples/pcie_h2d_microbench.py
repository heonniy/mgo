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
from pcie_host import GPUS, ROOT, affinity, write, topology, available_bytes
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
    source_bytes=a.source_gib*2**30 if a.source_gib else 16*EB
    source_experts=source_bytes//EB;assert source_experts*EB==source_bytes
    mappings={}
    if rank%2==0:
        pool=SharedPinnedMapping(root/f'node{node}.bin',source_bytes,node,create=True)
        if a.source_gib:
            # Equal source IDs have byte-identical BF16 payloads on both
            # nodes; row labels validate spread-address DMA outside timing.
            for index,row in enumerate(pool.tensor.view(source_experts,EB//2)):
                row.fill_(index%127+1)
        else:pool.tensor.fill_(node+1)
        mappings[node]=pool
    dist.barrier()
    receipts=[]
    for source_node in (0,1):
        if source_node not in mappings:
            mappings[source_node]=SharedPinnedMapping(root/f'node{source_node}.bin',source_bytes,source_node)
        if not a.source_gib or source_node==node:
            receipts.append(mappings[source_node].register())
        else:
            stat=mappings[source_node].path.stat()
            receipts.append(dict(numa_node=source_node,mapped_bytes=source_bytes,pinned=False,
                                 device=stat.st_dev,inode=stat.st_ino,
                                 reason='Remote control source registers only after the main local-source matrix'))
    write(a.out/f'host_rank{rank}.json',dict(rank=rank,gpu=GPUS[rank],affinity=sorted(os.sched_getaffinity(0)),sources=receipts,
          software=dict(torch=torch.__version__,cuda=torch.version.cuda,nccl=list(torch.cuda.nccl.version()))))
    rows={n:p.tensor.view(source_experts,EB//2) for n,p in mappings.items()}
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
        # compact changes the pinned footprint only; spread also removes
        # repeated reads of the same initial 144-MiB working set.
        base=((repeat+5)*211+sum(cell['counts'])*73)%source_experts
        prefix=sum(cell['counts'][:rank])
        indices=[index if a.source_layout=='compact' else (base+(prefix+index)*383)%source_experts for index in range(count)]
        start=release();comm=None
        if cell['overlap']:
            comm=dist.all_to_all_single(recv,send,group=nccl,async_op=True)
        for s in range(concurrency):
            with torch.cuda.stream(streams[s]):starts[s].record()
        for index in range(count):
            with torch.cuda.stream(streams[index%concurrency]):dest[index].copy_(rows[source_node][indices[index]],non_blocking=True)
        for s in range(concurrency):
            with torch.cuda.stream(streams[s]):ends[s].record()
        for s in range(concurrency):ends[s].synchronize()
        end=time.perf_counter_ns()
        service=max(starts[s].elapsed_time(ends[s]) for s in range(concurrency)) if count else 0
        if comm is not None:comm.wait();torch.cuda.synchronize()
        if repeat<0 and count:
            if a.source_gib:
                expected=torch.tensor([index%127+1 for index in indices],device='cuda',dtype=torch.bfloat16)[:,None]
                assert bool((dest[:count]==expected).all())
            else:assert torch.equal(dest[:count],torch.full_like(dest[:count],source_node+1))
        records.append(dict(cell=cell['name'],repeat=repeat,rank=rank,gpu=GPUS[rank],counts=cell['counts'],
                            streams=concurrency,remote=cell['remote'],overlap=cell['overlap'],
                            start_ns=start,ready_ns=end,bytes=count*EB,service_ms=service,
                            nccl_initialized=nccl is not None,source_bytes_per_node=source_bytes,
                            source_layout=a.source_layout,source_expert_indices=indices))
    cells=matrix()
    rng=random.Random(20261009)
    isolated=[c for c in cells if not c['overlap'] and not (a.source_gib and c['remote'])]
    for repeat in range(-5,30):
        order=isolated.copy();rng.shuffle(order)
        for cell in order:run(cell,repeat)
    if a.source_gib:
        # Main cells have exactly the real runtime's 54-GiB registration per
        # rank. Additional remote registration is outside all main timings.
        dist.barrier()
        remote_receipt=mappings[1-node].register()
        write(a.out/f'host_remote_control_rank{rank}.json',remote_receipt)
        controls=[c for c in cells if c['remote'] and not c['overlap']]
        for repeat in range(-5,30):
            order=controls.copy();rng.shuffle(order)
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
                source_bytes_per_node=records[0]['source_bytes_per_node'],source_layout=records[0]['source_layout'],
                source='one shared pinned pool per NUMA node; full-size main matrix registers only the local 54-GiB pool per rank, then adds remote registration outside main timing',
                isolated_nccl_never_initialized=True,cells=cells)
    write(out/'microbench_summary.json',result)
    write(out/'result.json',dict(status='PASS',cells=len(cells),timed_rank_samples=30*4*len(cells),pair_calibration_samples=len(pairs)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--rank',type=int);p.add_argument('--pool');p.add_argument('--port',type=int)
    p.add_argument('--source-gib',type=int,choices=(0,54),default=0);p.add_argument('--source-layout',choices=('compact','spread'),default='compact')
    a=p.parse_args()
    if a.rank is not None:worker(a);return
    a.out.mkdir(parents=True,exist_ok=False);write(a.out/'topology_before.json',topology())
    source_bytes=a.source_gib*2**30 if a.source_gib else 16*EB
    assert available_bytes()>=2*source_bytes+128*2**30,'shared source plus host headroom guard failed'
    assert shutil.disk_usage('/dev/shm').free>=2*source_bytes
    write(a.out/'config.json',dict(source_bytes_per_node=source_bytes,unique_source_bytes=2*source_bytes,
          source_layout=a.source_layout,shared_pairs=[[0,1],[4,5]],seed=20261009,
          argv=sys.argv,conda_prefix=sys.prefix,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PKG.parent,text=True).strip(),
          spread_address_rule='base=((repeat+5)*211+M*73)%6144; source=(base+global_fetch_index*383)%6144; identical global source IDs for equal-M quota arms',
          registration='54-GiB main cells register only the local shared pool per rank; remote controls register the other pool afterwards; 144-MiB legacy cells register both initially; no private payload copies',
          source_hash=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
    pool=Path('/dev/shm')/f'esjung_pcie_micro_{os.getpid()}'
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    processes=[];logs=[]
    with gpu_experiment('PCIe H2D microbenchmark and pair calibration'):
        try:
            for rank in range(4):
                log=(a.out/f'worker_rank{rank}.log').open('w');logs.append(log)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(GPUS[rank]),OMP_NUM_THREADS='1',NCCL_DEBUG='INFO',NCCL_DEBUG_FILE=str(a.out/f'nccl_rank{rank}.log'))
                processes.append(subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--out',str(a.out),'--rank',str(rank),'--pool',str(pool),'--port',str(port),'--source-gib',str(a.source_gib),'--source-layout',a.source_layout],env=env,stdout=log,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+600
            while any(x.poll() is None for x in processes):
                if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP file observed')
                if available_bytes()<128*2**30:raise RuntimeError('128-GiB host headroom was lost')
                if any(x.poll() not in (None,0) for x in processes):raise RuntimeError('microbenchmark worker failed; inspect rank logs')
                if time.monotonic()>deadline:raise TimeoutError('microbenchmark exceeded 600 seconds')
                time.sleep(.2)
            if any(x.returncode for x in processes):raise RuntimeError('microbenchmark worker failed')
        finally:
            # Worker cleanup must finish before the lease restores our burn.
            for process in processes:
                if process.poll() is None:process.terminate()
            for process in processes:
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            for log in logs:log.close()
            shutil.rmtree(pool,ignore_errors=True)
    summarize(a.out);write(a.out/'topology_after.json',topology())


if __name__=='__main__':main()
