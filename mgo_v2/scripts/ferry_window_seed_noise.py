"""Replay-noise reference for the W sweep: vary only the BR shuffle seed (CPU only).

A changed seed is a counterfactual (the live trace used seed 42); it is used
only to size how much the hit rate moves under an arbitrary, policy-neutral
perturbation of rank assignment, for comparison with the W effect.
"""
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ferry_window_sensitivity as f

SEEDS = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)
WS = (32, 128, 1024)


def job(args):
    capture, w = args
    tr, ranks = f.load(Path(capture)); counts = tr['counts']
    gt = f.gate_trace(ranks, ('window', w)); out = []
    for pct in f.CAPS:
        for s in SEEDS:
            m, _, tok = f.replay_tw(counts, gt, np.array(f.CAPS[pct], np.int32), s)
            d = m[f.L:].sum(0); t = tok[f.L:].sum(0)
            out.append(dict(capacity_pct=pct, window=w, seed=s, hit_rate=d[0] / (d[0] + d[1]), token_hit_rate=t[0] / t[1], miss=int(d[1])))
    return out


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    output = Path(sys.argv[1]); captures = sys.argv[2:]
    jobs = [(c, w) for c in captures for w in WS]
    with ProcessPoolExecutor(9) as ex: res = list(ex.map(job, jobs))
    with (output / 'SEED_NOISE.csv').open('w') as fh:
        wr = csv.DictWriter(fh, ['local_batch', 'capacity_pct', 'window', 'seed', 'hit_rate', 'token_hit_rate', 'miss']); wr.writeheader()
        for (c, _), rows in zip(jobs, res):
            b = int(Path(c).name.split('_')[1][1:])
            for r in rows: wr.writerow(dict(local_batch=b, **r))


if __name__ == '__main__':
    main()
