"""Why does a large random pinned source slow GPU4-7 DMA (0.20 -> 0.23 ms per 9 MiB)?

Sources per rank (all pinned, 9 MiB experts):
  rand_0.14g / rand_2g / rand_8g / rand_24g : random expert from a pool of that size (cudaHostAlloc)
  seq_24g  : 24 GiB pool, consecutive experts (walks the pool in order)
  thp_24g  : 24 GiB anonymous mmap with MADV_HUGEPAGE, cudaHostRegister'ed, random experts
Modes: all 8 GPUs concurrent with FAST quota [8]*4+[9]*4 (batch submit), and GPU4 solo (9 copies).
"""
import ctypes, json, mmap, os, random, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = 8
EXPERT = 9 * 2**20
REPS = int(os.environ.get('REPS', '30'))
SOURCES = ['rand_0.14g', 'rand_2g', 'rand_8g', 'rand_24g', 'seq_24g', 'thp_24g']
FAST = [8] * 4 + [9] * 4


def spin_until(t):
    while time.time_ns() < t:
        pass


def thp_pinned(nbytes):
    import torch
    m = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    m.madvise(mmap.MADV_HUGEPAGE)
    buf = np.frombuffer(m, dtype=np.uint8)
    buf[::4096] = 1  # fault in (THP where possible)
    ptr = buf.ctypes.data
    err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
    assert int(err) == 0, err
    t = torch.from_numpy(buf).view(torch.bfloat16)
    return m, buf, t


def worker(rank, conn, barrier, start_at):
    import torch
    os.sched_setaffinity(0, {rank * 24})
    torch.cuda.set_device(rank)
    n = EXPERT // 2
    pools = {}
    for name in ('rand_0.14g', 'rand_2g', 'rand_8g', 'rand_24g'):
        gib = float(name.split('_')[1][:-1])
        k = max(16, int(gib * 2**30 // EXPERT))
        pools[name] = torch.empty((k, n), dtype=torch.bfloat16, pin_memory=True)
    pools['seq_24g'] = pools['rand_24g']
    keep = thp_pinned(24 * 2**30)
    flat = keep[2]
    pools['thp_24g'] = flat[:(flat.numel() // n) * n].view(-1, n)
    for p in pools.values():
        p[:, :1].fill_(1)
    dst = torch.empty((12, n), dtype=torch.bfloat16, device='cuda')
    stream = torch.cuda.Stream()
    rng = random.Random(77 + rank)
    cursor = 0
    torch.cuda.synchronize()
    conn.send('ready')
    while True:
        msg = conn.recv()
        if msg is None:
            break
        q, source = msg
        q = q[rank]
        pool = pools[source]
        if source == 'seq_24g':
            idx = [(cursor + i) % pool.shape[0] for i in range(q)]; cursor += q
        else:
            idx = [rng.randrange(pool.shape[0]) for _ in range(q)]
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(q)]
        barrier.wait()
        spin_until(start_at.value)
        with torch.cuda.stream(stream):
            for i in range(q):
                evs[i][0].record(stream); dst[i].copy_(pool[idx[i]], non_blocking=True); evs[i][1].record(stream)
        stream.synchronize()
        barrier.wait()
        conn.send([a.elapsed_time(b) for a, b in evs])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'pool_cause_bench.json')
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(W + 1); start_at = ctx.Value('q', 0)
    pipes, procs = [], []
    for r in range(W):
        a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(r, b, barrier, start_at)); p.start()
        pipes.append(a); procs.append(p)
    for a in pipes:
        assert a.recv() == 'ready'
    print('ready; AnonHugePages:', [l for l in open('/proc/meminfo') if 'AnonHuge' in l][0].strip(), flush=True)

    def burst(q, source):
        for a in pipes:
            a.send((q, source))
        start_at.value = time.time_ns() + 3_000_000
        barrier.wait(); barrier.wait()
        return [a.recv() for a in pipes]

    solo = [0, 0, 0, 0, 9, 0, 0, 0]
    jobs = [(s, m) for s in SOURCES for m in ('all8', 'solo4')] * 2
    random.Random(11).shuffle(jobs)
    for s in SOURCES:
        for _ in range(5):
            burst(FAST, s)
    samples = {f'{s}_{m}': [] for s in SOURCES for m in ('all8', 'solo4')}
    for s, m in jobs:
        for _ in range(REPS // 2):
            samples[f'{s}_{m}'].append(burst(FAST if m == 'all8' else solo, s))
    for a in pipes:
        a.send(None)
    for p in procs:
        p.join()
    summary = {}
    for k, ss in samples.items():
        A = [x for s in ss for r in range(4) for x in s[r]]
        B = [x for s in ss for r in range(4, 8) for x in s[r]]
        summary[k] = dict(dma_A=float(np.mean(A)) if A else None, dma_B=float(np.mean(B)))
    json.dump(dict(summary=summary, samples=samples), open(out, 'w'))
    print(f"{'source':12s} {'all8 A':>7s} {'all8 B':>7s} {'A/B':>5s} {'solo GPU4':>9s}")
    for s in SOURCES:
        a, so = summary[f'{s}_all8'], summary[f'{s}_solo4']
        print(f"{s:12s} {a['dma_A']:7.3f} {a['dma_B']:7.3f} {a['dma_A'] / a['dma_B']:5.2f} {so['dma_B']:9.3f}")


if __name__ == '__main__':
    main()
