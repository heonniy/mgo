"""Validate real shared pages, CUDA host registration and paired GPU DMA."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from pcie_host import GPUS, NODES, affinity, write


def worker(rank, output, pool_root, full=False):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(GPUS[rank])
    os.sched_setaffinity(0, affinity(rank))
    import torch
    from mgo_v2.numa_shared_source import SharedPinnedMapping
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    size = 54 * 2**30 if full else 4 * 9437184
    sample_bytes = 9437184 if full else size
    target = torch.empty(sample_bytes // 2, device='cuda', dtype=torch.bfloat16)
    node = NODES[rank]
    path = pool_root / f'node{node}.bin'
    ready = pool_root / f'node{node}.ready'
    if rank % 2 == 0:
        pool = SharedPinnedMapping(path, size, node, create=True)
        pool.tensor.fill_((node + 1) / 8)
        ready.touch()
    else:
        deadline = time.monotonic() + 600
        while not ready.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('shared source creator')
            time.sleep(0.02)
        pool = SharedPinnedMapping(path, size, node)
    receipt = pool.register()
    write(output / f'ready_rank{rank}.json', receipt)
    deadline = time.monotonic() + 60
    while not (output / 'GO').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('paired DMA release')
        time.sleep(0.01)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    target.copy_(pool.tensor[:sample_bytes//2], non_blocking=True)
    end.record()
    end.synchronize()
    actual = target.cpu()
    assert torch.equal(actual, pool.tensor[:sample_bytes//2])
    if full:
        for offset in (size//4, size//2, size*3//4, size-sample_bytes):
            source=pool.tensor[offset//2:(offset+sample_bytes)//2]
            target.copy_(source,non_blocking=True);torch.cuda.synchronize()
            assert torch.equal(target.cpu(),source)
    receipt.update(status='PASS', rank=rank, physical_gpu=GPUS[rank],
                   h2d_ms=start.elapsed_time(end),
                   source_sha256=hashlib.sha256(pool.tensor[:sample_bytes//2].view(torch.uint8).numpy()).hexdigest(),
                   destination_sha256=hashlib.sha256(actual.view(torch.uint8).numpy()).hexdigest(),
                   unregister_verified=False,full_source=full,validated_samples=5 if full else 1,sample_bytes=sample_bytes)
    pool.unregister()
    receipt['unregister_verified'] = not pool.registered
    write(output / f'result_rank{rank}.json', receipt)


def main(args):
    if args.rank is not None:
        return worker(args.rank, args.output, args.pool_root,args.full)
    if args.full:
        from pcie_host import available_bytes
        assert available_bytes() >= (108+128)*2**30,'128-GiB global headroom guard'
    args.output.mkdir(parents=True, exist_ok=False)
    args.pool_root.mkdir(parents=True, exist_ok=False)
    children = []
    logs = []
    try:
        for rank in range(4):
            log = (args.output / f'rank{rank}.log').open('w')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPUS[rank]), OMP_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            children.append(subprocess.Popen([sys.executable, __file__, '--rank', str(rank),
                         '--output', str(args.output), '--pool-root', str(args.pool_root)]+(['--full'] if args.full else []),
                         env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + 600
        while not all((args.output / f'ready_rank{rank}.json').exists() for rank in range(4)):
            if any(child.poll() is not None for child in children):
                raise RuntimeError('Shared-source worker failed; see rank logs')
            if time.monotonic() > deadline:
                raise TimeoutError('shared source preflight')
            time.sleep(0.1)
        (args.output / 'GO').touch()
        if args.full:
            from pcie_host import available_bytes
            assert available_bytes()>=128*2**30,'128-GiB post-pinning headroom guard'
        for child in children:
            assert child.wait(timeout=60) == 0, 'DMA worker failed; see rank logs'
        rows = [json.loads((args.output / f'result_rank{rank}.json').read_text()) for rank in range(4)]
        for left, right in ((0, 1), (2, 3)):
            assert (rows[left]['device'], rows[left]['inode']) == (rows[right]['device'], rows[right]['inode'])
            assert rows[left]['source_sha256'] == rows[right]['source_sha256']
        assert rows[0]['inode'] != rows[2]['inode']
        write(args.output / 'result.json', dict(status='PASS', ranks=rows,
              unique_physical_source_bytes=2 * rows[0]['mapped_bytes'],
              total_rank_mapped_source_bytes=4 * rows[0]['mapped_bytes']))
        print('PASS: shared inode per NUMA pair, local sampled pages, pinned recognition, simultaneous DMA, exact bytes and unregister')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        shutil.rmtree(args.pool_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool-root', type=Path, required=True)
    parser.add_argument('--rank', type=int, choices=range(4))
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args()
    if args.rank is not None:main(args)
    else:
        from pcie_gpu_occupancy import gpu_experiment
        with gpu_experiment('full shared pinning validation' if args.full else 'small shared pinning validation'):main(args)
