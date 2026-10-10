"""Measure 8-GPU concurrent 9 MiB pinned H2D with per-rank (unbalanced) copy counts.

One persistent process per GPU, rank-private pinned source buffers, one stream.
Each burst: barrier -> common future start time (spin) -> q_r async copies ->
CUDA-event elapsed per rank. Cases are interleaved in shuffled order per round.
Only H2D service is measured: no model, NCCL or expert compute.
"""
import json, os, random, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from h2d_model import predict_rank_ms

W = 8
EXPERT = 9 * 2**20
MAXQ = 20
WARMUP, REPS, ROUNDS = 20, 60, 3

CASES = {
    'bal8': [8] * 8, 'bal9': [9] * 8, 'bal10': [10] * 8,
    'm12_near': [2, 2, 2, 2, 1, 1, 1, 1], 'm12_fast': [1, 1, 1, 1, 2, 2, 2, 2],
    'm12_unbal': [1, 1, 1, 0, 3, 2, 2, 2],
    'm68_near': [9] * 4 + [8] * 4, 'm68_fast': [8] * 4 + [9] * 4,
    'm68_ph': [7] * 4 + [10] * 4, 'm68_6_11': [6] * 4 + [11] * 4, 'm68_rev': [10] * 4 + [7] * 4,
    'm68_mix': [8, 8, 7, 7, 10, 10, 9, 9],
    'm74_near': [10, 10, 9, 9, 9, 9, 9, 9], 'm74_fast': [9, 9, 9, 9, 10, 9, 9, 10],
    'm74_ph': [8, 8, 8, 7, 11, 11, 11, 10],
    'm80_ph': [8] * 4 + [12] * 4,
}


if os.environ.get('GRID'):
    CASES = {f'A{a}_B{b}': [a] * 4 + [b] * 4 for a in range(4, 13) for b in range(6, 16)}
    REPS, ROUNDS = 20, 2

if os.environ.get('VALIDATE'):
    _t = json.load(open(os.path.join(HERE, 'SPLIT_LOOKUP.json')))['quota_lut']
    CASES = {}
    for _m in list(range(8, 128, 4)) + [69, 71, 73, 75, 77]:
        CASES[f'm{_m}_near'] = [_m // W + (r < _m % W) for r in range(W)]
        CASES[f'm{_m}_fast'] = list(_t['NEAR_FAST'][_m])
        CASES[f'm{_m}_split'] = list(_t['NEAR_SPLIT'][_m])
    REPS, ROUNDS = 20, 2


def worker(rank, conn, barrier, start_at):
    import torch
    os.sched_setaffinity(0, {rank * 24})
    torch.cuda.set_device(rank)
    n = EXPERT // 2
    src = [torch.empty(n, dtype=torch.bfloat16).pin_memory() for _ in range(MAXQ)]
    for s in src:
        s.fill_(1)
    dst = [torch.empty(n, dtype=torch.bfloat16, device='cuda') for _ in range(MAXQ)]
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    conn.send('ready')
    while True:
        msg = conn.recv()
        if msg is None:
            break
        q = msg[rank]
        barrier.wait()
        t_start = start_at.value
        while time.time_ns() < t_start:
            pass
        launch = time.time_ns()
        ms = 0.0
        if q:
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                a.record(stream)
                for i in range(q):
                    dst[i].copy_(src[i], non_blocking=True)
                b.record(stream)
            b.synchronize()
            ms = a.elapsed_time(b)
        done = time.time_ns()
        barrier.wait()
        conn.send((ms, launch, done))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'unbalanced_bench.json')
    ctx = mp.get_context('spawn')
    barrier = ctx.Barrier(W + 1)
    start_at = ctx.Value('q', 0)
    pipes, procs = [], []
    for r in range(W):
        a, b = ctx.Pipe()
        p = ctx.Process(target=worker, args=(r, b, barrier, start_at)); p.start()
        pipes.append(a); procs.append(p)
    for a in pipes:
        assert a.recv() == 'ready'

    def burst(q):
        for a in pipes:
            a.send(q)
        start_at.value = time.time_ns() + 3_000_000
        barrier.wait()
        barrier.wait()
        return [a.recv() for a in pipes]

    for _ in range(WARMUP):
        burst([8] * 8)
    samples = {k: [] for k in CASES}
    rng = random.Random(20261010)
    for rnd in range(ROUNDS):
        order = list(CASES)
        rng.shuffle(order)
        for k in order:
            for _ in range(REPS):
                res = burst(CASES[k])
                ms = [x[0] for x in res]
                launches = [x[1] for r, x in enumerate(res) if CASES[k][r] > 0]
                skew = (max(launches) - min(launches)) / 1e6
                samples[k].append(dict(round=rnd, ms=ms, skew_ms=skew))
        print(f'round {rnd} done', flush=True)
    for a in pipes:
        a.send(None)
    for p in procs:
        p.join()

    summary = {}
    for k, q in CASES.items():
        arr = np.array([s['ms'] for s in samples[k]])
        med = np.median(arr, 0)
        pred = predict_rank_ms(q)
        mx = arr.max(1)
        summary[k] = dict(quota=q, rank_median_ms=med.round(4).tolist(), pred_rank_ms=pred.round(4).tolist(),
                          max_rank_median_ms=float(np.median(mx)), max_rank_p10_p90=[float(np.percentile(mx, 10)), float(np.percentile(mx, 90))],
                          pred_max_ms=float(pred.max()),
                          per_round_max_median=[float(np.median(mx[[s['round'] == r for s in samples[k]]])) for r in range(ROUNDS)],
                          skew_ms_max=float(max(s['skew_ms'] for s in samples[k])))
    json.dump(dict(cases=summary, samples=samples, warmup=WARMUP, reps=REPS, rounds=ROUNDS), open(out, 'w'))
    print(f"{'case':10s} {'quota':28s} {'meas_max':>8s} {'pred_max':>8s}  A_med  B_med  rounds")
    for k, s in summary.items():
        A = np.mean([s['rank_median_ms'][i] for i in range(4) if s['quota'][i]]) if any(s['quota'][:4]) else 0
        Bm = np.mean([s['rank_median_ms'][i] for i in range(4, 8) if s['quota'][i]])
        print(f"{k:10s} {str(s['quota']):28s} {s['max_rank_median_ms']:8.3f} {s['pred_max_ms']:8.3f}  {A:.3f}  {Bm:.3f}  "
              + ' '.join(f'{v:.3f}' for v in s['per_round_max_median']) + f"  skew<={s['skew_ms_max']:.2f}")


if __name__ == '__main__':
    main()
