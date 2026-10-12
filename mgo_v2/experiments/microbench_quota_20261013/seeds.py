"""Replicate BALANCED_RANDOM_NEAR (code 25) remainder draws offline and score salts with the fluid model."""
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / 'scripts')); sys.path.insert(0, str(PKG))
import mgo_v2  # noqa: F401,E402
from env_offload_policy import seed_rng  # noqa: E402  same compiled RNG seeding as the runtime


@njit
def _perm(world):
    return np.random.permutation(world)


def quota(salt, event, m, world=8):
    seed_rng(salt * 1000003 + event)
    q = np.full(world, m // world, np.int64)
    for r in _perm(world)[:m % world]:
        q[r] += 1
    return q


def qwen_events(job):
    d = json.loads((Path(job) / 'generation_diagnostic_rank0.json').read_text())
    return [(e['event_index'], sum(e['rank_fetches']), e['rank_fetches']) for e in d['decode_placement_events']]



def imbalance(q, rates, cache):
    key = tuple(int(x) for x in q)
    if key not in cache:
        from fluid import finish_times
        t = finish_times(key, rates)
        cache[key] = (max(t) - sum(t) / len(t), max(t))
    return cache[key]


def score(events, salt, rates, cache):
    """Model sum over decode events of (max - mean) finish time and of the makespan, for one salt."""
    imb = mk = 0.0
    for event, m, _ in events:
        a, b = imbalance(quota(salt, event, m), rates, cache)
        imb += a; mk += b
    return imb, mk


def search(events, payload, salts=range(1, 2001)):
    from fluid import load_rates
    from build_tables import FAST_ORDER, balanced
    rates = load_rates(payload); cache = {}
    scores = {s: score(events, s, rates, cache) for s in salts}
    ref = lambda f: tuple(map(sum, zip(*[imbalance(f(m), rates, cache) for _, m, _ in events])))
    worst = max(scores, key=lambda s: scores[s][0]); best = min(scores, key=lambda s: scores[s][0])
    imbs = sorted(v[0] for v in scores.values())
    return dict(payload=payload, events=len(events), salts=[min(salts), max(salts)],
                worst_salt=worst, worst=scores[worst], best_salt=best, best=scores[best],
                median_imbalance=imbs[len(imbs) // 2],
                near=ref(lambda m: balanced(m, range(m % 8))), fast=ref(lambda m: balanced(m, FAST_ORDER[:m % 8])),
                units='sum over events of model finish times, in expert copies per GiB/s')
