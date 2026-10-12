"""Background H2D contention injector: paced pinned-host -> GPU copies on one GPU.

Creates real host->device contention on a chosen GPU group (induced contention, not an emulated delay).
Each copy is 9 MiB from a random offset of a large pinned pool, the same shape as expert fetches.
Pacing: one copy in flight; after it completes, sleep so that the long-run rate is --gibps.
--gibps 0 means back-to-back (maximum). --streams k keeps k copies in flight on k streams
(back-to-back only), to take a larger share of the link. The hog exits on SIGTERM or when --seconds elapse.
"""
import argparse
import json
import os
import random
import signal
import time

import torch

EXPERT = 9 * 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--gibps', type=float, required=True, help='target rate; 0 = back-to-back')
    p.add_argument('--pool-gib', type=float, default=8.0)
    p.add_argument('--seconds', type=float, default=36000.0)
    p.add_argument('--cpu', type=int)
    p.add_argument('--report')
    p.add_argument('--streams', type=int, default=1)
    a = p.parse_args()
    if a.cpu is not None:
        os.sched_setaffinity(0, {a.cpu})
    torch.cuda.set_device(a.gpu)
    n = EXPERT // 2
    rows = int(a.pool_gib * 2**30 // EXPERT)
    pool = torch.empty((rows, n), dtype=torch.bfloat16, pin_memory=True)
    pool[:, :1].fill_(1)
    assert a.streams == 1 or a.gibps == 0, 'multi-stream mode is back-to-back only'
    dst = torch.empty((a.streams, n), dtype=torch.bfloat16, device='cuda')
    streams = [torch.cuda.Stream() for _ in range(a.streams)]
    stream = streams[0]
    rng = random.Random(a.gpu)
    period = EXPERT / (a.gibps * 2**30) if a.gibps > 0 else 0.0
    stop = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    start = time.perf_counter(); copies = 0
    while not stop['flag'] and time.perf_counter() - start < a.seconds:
        t0 = time.perf_counter()
        for i, st in enumerate(streams):
            with torch.cuda.stream(st):
                for _ in range(4 if a.streams > 1 else 1):
                    dst[i].copy_(pool[rng.randrange(rows)], non_blocking=True)
                    copies += 1
        for st in streams:
            st.synchronize()
        rest = period - (time.perf_counter() - t0)
        if rest > 0:
            time.sleep(rest)
    elapsed = time.perf_counter() - start
    if a.report:
        with open(a.report, 'w') as f:
            json.dump(dict(gpu=a.gpu, target_gibps=a.gibps, copies=copies, seconds=elapsed,
                           achieved_gibps=copies * EXPERT / 2**30 / elapsed), f)


if __name__ == '__main__':
    main()
