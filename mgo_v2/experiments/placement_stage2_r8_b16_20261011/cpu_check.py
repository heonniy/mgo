"""CPU check of stage-2 placement controls under the FAST quota (ShareGPT route pool proxy, R8/B16).

Per decode event: critical-rank expert rows / mean (compute imbalance) and remote expert
rows (token rows routed to another rank, from the controller row[29]).
"""
import json, sys
import numpy as np
R = '/home/hwlee/mgo-haq/mgo_v2'
sys.path[:0] = ['/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps', R, R + '/scripts']
import mgo_v2.controller  # noqa: F401
from env_offload_policy import Policy

POOL = '/home/hwlee/mgo-results/ca_stress_workload_search_20261004/pool/ShareGPT/'
LUT = np.asarray(json.load(open(R + '/experiments/pcie_quota_r8_b16_20261010/LOOKUP.json'))['quota_lut']['NEAR_FAST'], np.int64)
W, B, E, L, WARM, STEPS = 8, 16, 128, 48, 8, 32
CAPS = [c - 2 for c in [231, 231, 231, 230, 230, 230, 230, 230]]


def run(code, seed=0):
    SEL = np.load(POOL + 'decode_selected.npy', mmap_mode='r'); WT = np.load(POOL + 'decode_weights.npy', mmap_mode='r')
    PR = np.load(POOL + 'decode_router.npy', mmap_mode='r')
    req = np.sort(np.random.default_rng(seed).choice(SEL.shape[2], W * B, replace=False))
    pol = Policy(CAPS, np.zeros((L, E, E), np.float32), False, 0, 42)
    org = np.repeat(np.arange(W), B).astype(np.int64); hist = [[] for _ in range(L)]; ev = 0
    imb, remote, total = [], 0, 0
    for s in range(WARM + STEPS):
        if s == WARM:
            pol.policy = code; pol.quota_lut = np.ascontiguousarray(LUT)
        sel = np.asarray(SEL[s][:, req]).astype(np.int16); w = np.asarray(WT[s][:, req]); p = np.asarray(PR[s][:, req])
        for l in range(L):
            hist[l].append(p[l]); g = np.concatenate(hist[l][-8:])[-128:].mean(0).astype(np.float32)
            _, eff, _, lens, dest, fetches, row = pol.apply(ev, sel[l], w[l], org, g, np.zeros((E, W), np.int32)); ev += 1
            if s < WARM:
                continue
            mask = np.arange(eff.shape[1])[None] < lens[:, None]
            rows = np.bincount(dest[mask].astype(np.int64), minlength=W)
            imb.append(rows.max() / rows.mean()); remote += row[29]; total += row[28]
    return float(np.mean(imb)), float(remote / total)


if __name__ == '__main__':
    for name, code in (('FAST+Near', 12), ('FAST+Random', 23), ('FAST+Worst', 22)):
        i, r = run(code)
        print(f'{name:12s} critical/mean rows {i:.3f}   remote row fraction {r:.3f}')
