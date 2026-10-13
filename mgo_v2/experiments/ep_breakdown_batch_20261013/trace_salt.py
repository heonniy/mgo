"""Worst-salt search for BR_SALT (code 26) on a recorded cell trace (per-event origin-rank x expert demand + gates)."""
import sys, os, json, numpy as np
R = '/home/hwlee/mgo-haq/mgo_v2'
sys.path[:0] = ['/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps', R, R + '/scripts', R + '/examples']
import mgo_v2.controller
from env_offload_policy import Policy
TR = np.load(sys.argv[1]); H = TR['histograms']; G = TR['gates']; W = H.shape[1]; E = 128
CAPS = [459, 459, 459, 458]
def tokens(hist):
    sel = []; org = []
    for r in range(W):
        c = hist[r]; n = int(c.sum()) // 8
        if n == 0: continue
        occ = np.repeat(np.arange(E), c)            # experts grouped, each count <= n
        s = occ.reshape(8, n).T.astype(np.int64).copy()   # occurrence i -> token i % n, slot i // n
        sel.append(s); org.append(np.full(n, r, np.int64))
    return np.concatenate(sel), np.concatenate(org)
CACHE = {}
def ev_tokens(i):
    if i not in CACHE: CACHE[i] = tokens(H[i])
    return CACHE[i]
def run(code, salt):
    os.environ['MGO_RANDOM_SALT'] = str(salt)
    pol = Policy(CAPS, np.zeros((48, 128, 128), np.float32), False, 0, 42)
    T = 0.0
    for i in range(H.shape[0]):
        if i == 48: pol.policy = code
        sel, org = ev_tokens(i)
        _, eff, _, lengths, dest, fetches, _ = pol.apply(i, sel, np.full(sel.shape, 0.125, np.float32), org, G[i], np.zeros((128, W), np.int32))
        if i < 48: continue
        q = np.bincount([f[0] for f in fetches], minlength=W)
        L = np.arange(eff.shape[1])[None, :] < lengths[:, None]
        er = np.bincount((dest.astype(np.int64) * 128 + eff.astype(np.int64))[L], minlength=W * 128).reshape(W, 128)
        T += (q * 0.18 + (er > 0).sum(1) * 0.2 + er.sum(1) * 0.0023).max()
    return T / ((H.shape[0] - 48) / 48)
if __name__ == '__main__':
    salts = [int(x) for x in sys.argv[2].split(',')] if sys.argv[2] != 'ref' else []
    if sys.argv[2] == 'ref':
        for name, code in (('BR', 0), ('LA_CA_NEAR', 7)): print(json.dumps(dict(salt=name, ms=run(code, 0))), flush=True)
    for s in salts: print(json.dumps(dict(salt=s, ms=run(26, s))), flush=True)
