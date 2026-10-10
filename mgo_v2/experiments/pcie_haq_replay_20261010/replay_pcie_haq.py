"""CPU replay: R8/B16/C30 decode miss-quota policies with a PCIe-group H2D model.

Routes: Qwen3 ShareGPT BR route pool (proxy; not the exact R8 L512 workload).
Policies all use the production controller step (Near assignment inside quota);
only the per-layer quota vector differs.
"""
import functools, json, sys, time
from multiprocessing import Pool
import numpy as np

R = '/home/hwlee/mgo-haq/mgo_v2'
HERE = '/home/hwlee/mgo-haq/mgo_v2/experiments/pcie_haq_replay_20261010'
sys.path[:0] = [HERE, '/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps', R, R + '/scripts']
import mgo_v2.controller  # noqa: F401  (package init before script adapters)
from env_offload_policy_x import Policy
from h2d_model import predict_rank_ms

POOL = '/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool/ShareGPT/'
LUT = json.load(open(R + '/experiments/pcie_quota_r8_b16_20261010/LOOKUP.json'))['quota_lut']
W, B, E, L = 8, 16, 128, 48
CAPS = [c - 2 for c in [231, 231, 231, 230, 230, 230, 230, 230]]
WARM, STEPS = 8, 64
C_SERIAL, C_GROUP = 0.2, 0.03


@functools.lru_cache(maxsize=None)
def h2d(q):
    return predict_rank_ms(q)


def objective(q, hits, mode):
    t = h2d(tuple(q))
    if mode == 'h2d':
        return t.max(), t.sum()
    if mode == 'serial':
        c = t + (hits + q) * C_SERIAL
    else:  # grouped: hits overlap H2D, misses grouped afterwards
        c = np.maximum(t, hits * C_GROUP) + q * C_GROUP
    return c.max(), c.sum()


def greedy_quota(m, hits, mode):
    q = np.zeros(W, np.int64)
    for _ in range(m):
        best = None
        for r in range(W):
            q[r] += 1
            key = objective(q, hits, mode) + (r,)
            q[r] -= 1
            if best is None or key < best:
                best = key
        q[best[2]] += 1
    return q


def run(args):
    name, seed = args
    SEL = np.load(POOL + 'decode_selected.npy', mmap_mode='r')
    WT = np.load(POOL + 'decode_weights.npy', mmap_mode='r')
    PR = np.load(POOL + 'decode_router.npy', mmap_mode='r')
    req = np.sort(np.random.default_rng(seed).choice(SEL.shape[2], W * B, replace=False))
    sim = np.zeros((L, E, E), np.float32)
    code = {'NEAR': 7, 'FAST': 11, 'PCIE_LUT': 11, 'HAQ': 16}.get(name, 20)
    lut = None
    if name == 'FAST':
        lut = np.asarray(LUT['NEAR_FAST'], np.int64)
    elif name == 'PCIE_LUT':
        lut = np.asarray(LUT['NEAR_PCIE'], np.int64)
    pol = Policy(CAPS, sim, False, 0, 42, quota_lut=None)  # BR warmup
    org = np.repeat(np.arange(W), B).astype(np.int64)
    hist = [[] for _ in range(L)]
    ev = 0
    rows = []
    for s in range(WARM + STEPS):
        if s == WARM:
            pol.policy = code
            if lut is not None:
                pol.quota_lut = np.ascontiguousarray(lut)
            elif code == 20:
                pol.quota_lut = np.zeros((129, W), np.int64)
        sel = np.asarray(SEL[s][:, req]).astype(np.int16)
        w = np.asarray(WT[s][:, req])
        p = np.asarray(PR[s][:, req])
        for l in range(L):
            hist[l].append(p[l])
            g = np.concatenate(hist[l][-8:])[-128:].mean(0).astype(np.float32)
            active = np.zeros(E, bool)
            active[np.unique(sel[l])] = True
            key0 = l * E
            resident = pol.owner[key0:key0 + E] != 0
            m = int((active & ~resident).sum())
            hits = np.bincount(pol.primary[key0:key0 + E][active & resident], minlength=W).astype(np.int64)
            if s >= WARM and code == 20:
                mode = {'PH_H2D': 'h2d', 'PH_SERIAL': 'serial', 'PH_GROUP': 'grouped'}[name]
                pol.quota_lut[m] = greedy_quota(m, hits, mode)
            _, eff, _, lens, dest, fetches, _ = pol.apply(ev, sel[l], w[l], org, g, np.zeros((E, W), np.int32))
            ev += 1
            if s < WARM:
                continue
            q = np.bincount([f[0] for f in fetches], minlength=W).astype(np.int64)
            mask = np.arange(eff.shape[1])[None] < lens[:, None]
            ex = (np.bincount((dest.astype(np.int64) * E + eff)[mask], minlength=W * E).reshape(W, E) > 0).sum(1)
            rows.append((m, q, ex, hits))
    t = np.array([h2d(tuple(r[1])) for r in rows])          # (events, W)
    q = np.array([r[1] for r in rows]); ex = np.array([r[2] for r in rows])
    hx = ex - q
    per_tok = lambda c: float(c.max(1).sum() / STEPS)
    out = dict(policy=name, seed=seed, fetches=int(q.sum()), mean_m=float(np.mean([r[0] for r in rows])),
               copies_by_rank=q.sum(0).tolist(), experts_by_rank=ex.sum(0).tolist(),
               max_minus_min_mean=float((q.max(1) - q.min(1)).mean()),
               h2d_ms=per_tok(t),
               serial_c200_ms=per_tok(t + ex * 0.2), serial_c071_ms=per_tok(t + ex * 0.071),
               group_c02_ms=per_tok(np.maximum(t, hx * .02) + q * .02),
               group_c05_ms=per_tok(np.maximum(t, hx * .05) + q * .05))
    return out


if __name__ == '__main__':
    policies = ['NEAR', 'FAST', 'PCIE_LUT', 'HAQ', 'PH_H2D', 'PH_SERIAL', 'PH_GROUP']
    seeds = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [0, 1, 2]
    jobs = [(p, s) for s in seeds for p in policies]
    t0 = time.time()
    with Pool(len(jobs)) as pool:
        res = pool.map(run, jobs)
    json.dump(res, open(HERE + '/replay_results.json', 'w'), indent=1)
    print(f'{time.time() - t0:.0f}s')
    for r in res:
        print(r['policy'], r['seed'], r['fetches'], round(r['mean_m'], 1), round(r['max_minus_min_mean'], 2),
              *(round(r[k], 2) for k in ('h2d_ms', 'serial_c200_ms', 'serial_c071_ms', 'group_c02_ms', 'group_c05_ms')))
