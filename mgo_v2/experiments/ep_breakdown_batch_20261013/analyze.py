"""EP cost and rank-imbalance breakdown from the compact layer_breakdown.npz files.

Covers ../ep_breakdown_r4_serial_20261013/raw (B16: BR, LA_CA_NEAR, CA, CA_NATIVE, BR_SALT46) and ./raw (B32/B64: BR, LA_CA_NEAR).
Per run, decode layers only; times in ms per decode step (sum over the 63 decode steps / 63):
  compute_imbalance  = sum_l (max_r - mean_r) compute[l, r]
  load_imbalance     = sum_l (max_r - mean_r) (h2d_wait + compute)[l, r]      (dark bar)
  h2d_wait_imbalance = load_imbalance - compute_imbalance
  comm_cost          = sum_l [mean_r comm[l, r] - load_imbalance_l]           (light bar; comm spans net of waiting for the busiest rank)
  ep_cost            = load_imbalance + comm_cost
  comm_imbalance     = sum_l (max_r - mean_r) comm[l, r]
  controller         = mean_r sum_l placement_controller_cpu[l, r]
Volume ratios (mean over decode layers of max_r / mean_r): experts_ratio, rows_ratio; bytes_ratio is max_r/mean_r of each rank's
remote bytes sent over the run. TPOT is the run's uninstrumented target (repeat1.json); the breakdown comes from the separate
post-generation diagnostic pass. Full diagnostic JSONs (~50 MB each) are kept outside git under /home/hwlee/mgo-results/.
"""
import json, glob, os, statistics
from collections import defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ST = 63
def ratio(x):
    m = x.mean(1); ok = m > 0
    return float((x.max(1)[ok] / m[ok]).mean())
def metrics(path):
    z = np.load(f'{path}/layer_breakdown.npz'); comp, wait, comm = z['compute'], z['h2d_wait'], z['comm']
    own = comp + wait; oi = own.max(1) - own.mean(1); ci = comp.max(1) - comp.mean(1); f = lambda a: float(a) * 1000 / ST
    sent = z['sent_bytes'].astype(float)
    return dict(TPOT=json.load(open(f'{path}/repeat1.json'))['TPOT'], ep_cost=f(comm.mean(1).sum()), load_imbalance=f(oi.sum()),
                compute_imbalance=f(ci.sum()), h2d_wait_imbalance=f((oi - ci).sum()), comm_cost=f((comm.mean(1) - oi).sum()),
                comm_imbalance=f((comm.max(1) - comm.mean(1)).sum()), controller=f(z['controller'].mean(1).sum()),
                mean_compute=f(comp.mean(1).sum()), experts_ratio=ratio(z['expert_uses'].astype(float)),
                rows_ratio=ratio(z['token_expert_uses'].astype(float)), bytes_ratio=float(sent.max() / sent.mean()),
                remote_rows_per_layer=float(z['remote_expert_rows'].mean()), remote_packets_per_layer=float(z['remote_rank_packets'].mean()),
                remote_MiB_per_step=float(sent.sum() / 2**20 / ST))
runs = defaultdict(list)
for d in sorted(glob.glob(f'{HERE}/raw/b*_d[0-9]')) + sorted(glob.glob(f'{HERE}/../ep_breakdown_r4_serial_20261013/raw/c30_*_d[0-9]')):
    name = os.path.basename(d); pol = name.split('_', 1)[1].rsplit('_d', 1)[0]
    B = int(name[1:].split('_')[0]) if name.startswith('b') else 16
    runs[(B, pol)].append(metrics(d))
out = {}
for (B, pol), rs in sorted(runs.items()):
    out[f'B{B}/{pol}'] = dict(n=len(rs), **{k: dict(mean=statistics.mean(r[k] for r in rs), std=statistics.stdev(r[k] for r in rs) if len(rs) > 1 else 0.0,
                                                    values=[r[k] for r in rs]) for k in rs[0]})
json.dump(out, open(f'{HERE}/SUMMARY.json', 'w'), indent=1)
K = ('TPOT', 'ep_cost', 'load_imbalance', 'compute_imbalance', 'h2d_wait_imbalance', 'comm_cost', 'comm_imbalance', 'controller',
     'experts_ratio', 'rows_ratio', 'bytes_ratio', 'remote_rows_per_layer', 'remote_MiB_per_step')
print('| cell | n | ' + ' | '.join(K) + ' |'); print('|---' * (len(K) + 2) + '|')
for k, v in out.items():
    print(f"| {k} | {v['n']} | " + ' | '.join(f"{v[x]['mean']:.4f}" if x == 'TPOT' else (f"{v[x]['mean']:.3f}" if 'ratio' in x else f"{v[x]['mean']:.1f}") for x in K) + ' |')
