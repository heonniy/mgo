"""Calibrate induced H2D contention: hog rate on GPUs 0-3 vs per-copy DMA of the experiment-like workload.

Workload per burst (8 GPUs, as in-run): FAST quota [8]*4+[9]*4 copies of 9 MiB from a 24 GiB random pinned
pool, paced submission (0.2-0.5 ms start delay, 0.18 ms between copies). Hogs (scripts/h2d_contention_hog.py)
run as separate processes on GPUs 0-3 at a target rate. Reports per-copy DMA for GPUs 0-3 / 4-7 and the ratio.
"""
import json, os, random, subprocess, sys, time
import multiprocessing as mp
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
HOG = HERE.parents[1] / 'scripts/h2d_contention_hog.py'
PY = sys.executable
W, EXPERT, POOL_GIB = 8, 9 * 2**20, 24
QUOTA = [8] * 4 + [9] * 4
LEVELS = [int(x) for x in os.environ.get('LEVELS', '0,5,10,20,30,-1').split(',')]  # GiB/s; 0 = none, -1 = back-to-back
STREAMS = [int(x) for x in os.environ.get('STREAMS', '1').split(',')]  # hog streams for the -1 level
REPS = 40


def spin(t):
    while time.time_ns() < t:
        pass


def worker(rank, conn, barrier, start_at):
    import torch
    os.sched_setaffinity(0, {rank * 24})
    torch.cuda.set_device(rank)
    n = EXPERT // 2; rows = POOL_GIB * 2**30 // EXPERT
    pool = torch.empty((rows, n), dtype=torch.bfloat16, pin_memory=True); pool[:, :1].fill_(1)
    dst = torch.empty((12, n), dtype=torch.bfloat16, device='cuda')
    stream = torch.cuda.Stream(); rng = random.Random(100 + rank)
    torch.cuda.synchronize(); conn.send('ready')
    while True:
        msg = conn.recv()
        if msg is None:
            break
        q = QUOTA[rank]
        idx = [rng.randrange(rows) for _ in range(q)]
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(q)]
        delay = rng.uniform(0.2e6, 0.5e6)
        barrier.wait(); t0 = start_at.value; spin(t0 + int(delay))
        with torch.cuda.stream(stream):
            for i in range(q):
                if i:
                    spin(time.time_ns() + 180_000)
                ev[i][0].record(stream); dst[i].copy_(pool[idx[i]], non_blocking=True); ev[i][1].record(stream)
        stream.synchronize(); done = time.time_ns(); barrier.wait()
        conn.send(([a.elapsed_time(b) for a, b in ev], (done - t0) / 1e6))


def main():
    ctx = mp.get_context('spawn'); barrier = ctx.Barrier(W + 1); start_at = ctx.Value('q', 0)
    pipes, procs = [], []
    for r in range(W):
        a, b = ctx.Pipe(); p = ctx.Process(target=worker, args=(r, b, barrier, start_at)); p.start()
        pipes.append(a); procs.append(p)
    for a in pipes:
        assert a.recv() == 'ready'

    def burst():
        for a in pipes:
            a.send('go')
        start_at.value = time.time_ns() + 3_000_000
        barrier.wait(); barrier.wait()
        return [a.recv() for a in pipes]

    out = Path(os.environ.get('OUT', HERE / 'CALIBRATION.json'))
    results = json.loads(out.read_text()) if out.exists() else {}
    configs = [(lv, 1) for lv in LEVELS if lv >= 0] + [(-1, k) for k in STREAMS if -1 in LEVELS]
    for level, k in configs:
        tag = f'{level}' if level >= 0 else f'max_s{k}'
        hogs = []
        if level != 0:
            for g in range(4):
                rep = HERE / f'hog_report_{tag}_g{g}.json'
                hogs.append(subprocess.Popen([PY, str(HOG), '--gpu', str(g), '--gibps', str(max(level, 0)),
                                              '--streams', str(k), '--cpu', str(g * 24 + 20), '--report', str(rep)]))
            time.sleep(25)  # hog pinning + warm-up
        for _ in range(5):
            burst()
        dmaA, dmaB, span = [], [], []
        for _ in range(REPS):
            res = burst()
            for r, (d, w) in enumerate(res):
                (dmaA if r < 4 else dmaB).extend(d)
            span.append(max(w for _, w in res))
        for h in hogs:
            h.terminate()
        for h in hogs:
            h.wait(timeout=60)
        hog_rates = [json.load(open(HERE / f'hog_report_{tag}_g{g}.json'))['achieved_gibps'] for g in range(4)] if hogs else []
        results[tag] = dict(dma_A=float(np.mean(dmaA)), dma_B=float(np.mean(dmaB)),
                                   ratio=float(np.mean(dmaA) / np.mean(dmaB)), makespan=float(np.median(span)),
                                   hog_achieved_gibps=hog_rates)
        r = results[tag]
        print(f"hog {tag:>7}: dmaA {r['dma_A']:.3f} dmaB {r['dma_B']:.3f} "
              f"A/B {r['ratio']:.2f} makespan {r['makespan']:.2f} ms hog achieved {[round(x, 1) for x in hog_rates]}", flush=True)
    for a in pipes:
        a.send(None)
    for p in procs:
        p.join()
    out.write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
