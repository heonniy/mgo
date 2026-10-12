"""Start/stop induced H2D contention hogs (scripts/h2d_contention_hog.py) from an R8 job guard.

Hogs start after the guard's foreign-process check, so they are not mistaken for foreign work, and
they stop before the guard's final GPU-process check.
"""
import json
import subprocess
import time
from pathlib import Path

HOG = Path(__file__).resolve().parent / 'h2d_contention_hog.py'


def add_arguments(parser):
    parser.add_argument('--hog-gpus', default='', help='comma list of physical GPUs that get a contention hog')
    parser.add_argument('--hog-streams', type=int, default=1)
    parser.add_argument('--hog-gibps', type=float, default=0.0, help='0 = back-to-back')
    parser.add_argument('--hog-warmup-s', type=float, default=25.0)


def start(args, python, env, output):
    gpus = [int(g) for g in args.hog_gpus.split(',') if g != '']
    procs = []
    for g in gpus:
        procs.append(subprocess.Popen([python, str(HOG), '--gpu', str(g), '--gibps', str(args.hog_gibps),
                                       '--streams', str(args.hog_streams), '--cpu', str(g * 24 + 20),
                                       '--report', str(output / f'hog_g{g}.json')],
                                      env=env, stdout=subprocess.DEVNULL, stderr=(output / f'hog_g{g}.log').open('w')))
    if procs:
        time.sleep(args.hog_warmup_s)
        dead = [g for g, p in zip(gpus, procs) if p.poll() is not None]
        assert not dead, f'contention hog exited early on GPUs {dead}'
    return dict(gpus=gpus, streams=args.hog_streams, gibps=args.hog_gibps, procs=procs)


def stop(hogs, output):
    for p in hogs.get('procs', []):
        if p.poll() is None:
            p.terminate()
    for p in hogs.get('procs', []):
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill(); p.wait()
    rates = {}
    for g in hogs.get('gpus', []):
        f = output / f'hog_g{g}.json'
        rates[g] = json.loads(f.read_text())['achieved_gibps'] if f.exists() else None
    return dict(gpus=hogs.get('gpus', []), streams=hogs.get('streams'), gibps=hogs.get('gibps'), achieved_gibps=rates)
