"""Why is the in-run GPU0-3 / GPU4-7 DMA asymmetry ~1.08x instead of the burst 1.25x?

2x2 conditions, 8 GPUs, 9 MiB pinned H2D copies, per-rank copy counts from a quota:
  source: 'small' (16 reused pinned buffers) | 'large' (random experts from a POOL_GIB pinned pool)
  submit: 'batch' (all copies enqueued back to back at a common start)
          | 'paced' (random 0.2-0.5 ms start delay, then 0.18 ms between submissions,
            mimicking the in-run Python staging thread)
Per-copy DMA is CUDA-event begin->done on the copy stream (as in the runtime trace).
Makespan is host wall from the common start to the latest rank's completion.
"""
import json, os, random, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
W = 8
EXPERT = 9 * 2**20
POOL_GIB = int(os.environ.get('POOL_GIB', '24'))
SMALL = 16
MAXQ = 12
REPS = int(os.environ.get('REPS', '40'))
QUOTAS = {'bal9': [9] * 8, 'fast': [8] * 4 + [9] * 4, 'split': [7] * 4 + [10] * 4}
CONDS = [(s, m) for s in ('small', 'large') for m in ('batch', 'paced')]


def spin_until(t_ns):
    while time.time_ns() < t_ns:
        pass


def worker(rank, conn, barrier, start_at):
    import torch
    os.sched_setaffinity(0, {rank * 24})
    torch.cuda.set_device(rank)
    n = EXPERT // 2
    pool_n = POOL_GIB * 2**30 // EXPERT
    t0 = time.time()
    small = torch.empty((SMALL, n), dtype=torch.bfloat16, pin_memory=True)
    small.fill_(1)
    large = torch.empty((pool_n, n), dtype=torch.bfloat16, pin_memory=True)
    large[:, :1].fill_(1)  # touch rows lightly; pinned pages are already resident
    dst = torch.empty((MAXQ, n), dtype=torch.bfloat16, device='cuda')
    stream = torch.cuda.Stream()
    rng = random.Random(1000 + rank)
    torch.cuda.synchronize()
    conn.send(('ready', time.time() - t0, pool_n))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        q, source, mode = msg
        q = q[rank]
        idx = [rng.randrange(pool_n) for _ in range(q)] if source == 'large' else [i % SMALL for i in range(q)]
        evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(q)]
        delay = rng.uniform(0.2e6, 0.5e6) if mode == 'paced' else 0
        barrier.wait()
        t_start = start_at.value
        spin_until(t_start + int(delay))
        with torch.cuda.stream(stream):
            for i in range(q):
                if mode == 'paced' and i:
                    spin_until(time.time_ns() + 180_000)
                src = large[idx[i]] if source == 'large' else small[idx[i]]
                evs[i][0].record(stream)
                dst[i].copy_(src, non_blocking=True)
                evs[i][1].record(stream)
        stream.synchronize()
        done = time.time_ns()
        dma = [a.elapsed_time(b) for a, b in evs]
        barrier.wait()
        conn.send((dma, (done - t_start) / 1e6))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'dma_conditions_bench.json')
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(W + 1)
    start_at = ctx.Value('q', 0)
    pipes, procs = [], []
    for r in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(r, b, barrier, start_at)); p.start()
        pipes.append(a); procs.append(p)
    info = [a.recv() for a in pipes]
    print('ready; pin seconds', [round(x[1], 1) for x in info], 'pool experts', info[0][2], flush=True)

    def burst(q, source, mode):
        for a in pipes:
            a.send((q, source, mode))
        start_at.value = time.time_ns() + 3_000_000
        barrier.wait(); barrier.wait()
        return [a.recv() for a in pipes]

    for cond in CONDS:
        for _ in range(10):
            burst(QUOTAS['bal9'], *cond)
    jobs = [(c, k) for c in CONDS for k in QUOTAS] * 2
    random.Random(20261011).shuffle(jobs)
    samples = {f'{c[0]}_{c[1]}_{k}': [] for c in CONDS for k in QUOTAS}
    for (cond, k) in jobs:
        for _ in range(REPS // 2):
            res = burst(QUOTAS[k], *cond)
            samples[f'{cond[0]}_{cond[1]}_{k}'].append(dict(dma=[x[0] for x in res], wall=[x[1] for x in res]))
    for a in pipes:
        a.send(None)
    for p in procs:
        p.join()

    summary = {}
    for key, ss in samples.items():
        A = np.concatenate([np.concatenate([s['dma'][r] for r in range(4)]) for s in ss])
        B = np.concatenate([np.concatenate([s['dma'][r] for r in range(4, 8)]) for s in ss])
        wall = np.array([s['wall'] for s in ss])
        summary[key] = dict(dma_mean_A=float(A.mean()), dma_mean_B=float(B.mean()), ratio=float(A.mean() / B.mean()),
                            wallA_median=float(np.median(wall[:, :4].max(1))), wallB_median=float(np.median(wall[:, 4:].max(1))),
                            makespan_median=float(np.median(wall.max(1))), n=len(ss))
    json.dump(dict(summary=summary, samples=samples, pool_gib=POOL_GIB, reps=REPS), open(out, 'w'))
    print(f"{'condition':22s} {'dmaA':>6s} {'dmaB':>6s} {'A/B':>5s} {'wallA':>6s} {'wallB':>6s} {'makespan':>8s}")
    for key, s in summary.items():
        print(f"{key:22s} {s['dma_mean_A']:6.3f} {s['dma_mean_B']:6.3f} {s['ratio']:5.2f} {s['wallA_median']:6.3f} {s['wallB_median']:6.3f} {s['makespan_median']:8.3f}")


if __name__ == '__main__':
    main()
