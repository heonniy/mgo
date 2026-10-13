"""Summarize live FERRY Appendix-C TPOT runs (ftpot_* labels)."""
import csv, json, statistics, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
rows = []
for d in sorted(ROOT.glob('ftpot_*_v1')):
    pol = json.loads((d / 'ferry_policy.json').read_text()) if (d / 'ferry_policy.json').exists() else None
    st = json.loads((d / 'status.json').read_text())['status'] if (d / 'status.json').exists() else 'MISSING'
    if pol is None or st != 'PASS':
        rows.append(dict(label=d.name, status=st)); continue
    reps = []
    for r in (1, 2, 3):
        x = json.loads((d / f'repeat{r}.json').read_text())
        v = [json.loads((d / f'repeat{r}_rank{k}.json').read_text())['validation'] for k in range(4)]
        reps.append(dict(TTFT=x['TTFT'], TPOT=x['TPOT'], E2E=x['E2E'],
                         h2d=sum(q['scheduler']['copies'] for q in v), h2d_bytes=sum(q['scheduler']['bytes'] for q in v)))
    t = [r['TPOT'] for r in reps]
    rows.append(dict(label=d.name, status=st, **pol, tpot_median=statistics.median(t), tpot_min=min(t), tpot_max=max(t),
                     tpot_spread=(max(t) - min(t)) / statistics.mean(t), e2e_median=statistics.median(r['E2E'] for r in reps),
                     ttft_r2r3=statistics.mean(r['TTFT'] for r in reps[1:]),
                     h2d_fetches=reps[0]['h2d'], h2d_gib=reps[0]['h2d_bytes'] / 2**30,
                     h2d_identical_across_repeats=len({r['h2d'] for r in reps}) == 1,
                     tpot_repeats=' '.join(f'{x:.4f}' for x in t)))
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with (OUT / 'TPOT_ROWS.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if r['status'] == 'PASS']
NAME = {'lru': 'LRU', 'count': 'Selection Count', 'selected': 'Selected-only Gate', 'full': 'Full Gate Score (FERRY)'}
COL = {'full': '#2a78d6', 'selected': '#eb6834', 'count': '#1baf7a', 'lru': '#e87ba4'}
md = ['| Cache | B | Policy | W | TPOT median (s) | range (s) | vs Full | E2E median (s) | H2D fetches (prefill+decode, all ranks) | H2D GiB |',
      '|---|---|---|---:|---:|---|---:|---:|---:|---:|']
def find(p, b, s, w=128):
    return next((r for r in ok if r['cache_percent'] == p and r['local_batch'] == b and r['evict_score'] == s and r['gate_window'] == w), None)
for p in (30, 60):
    for b in (8, 16, 64):
        base = find(p, b, 'full')
        for s in ('full', 'selected', 'count', 'lru'):
            r = find(p, b, s)
            if r is None: continue
            rel = f"{100*(r['tpot_median']/base['tpot_median']-1):+.1f}%" if base else '-'
            md.append(f"| C{p} | B{b} | {NAME[s]} | 128 | {r['tpot_median']:.4f} | {r['tpot_min']:.4f}-{r['tpot_max']:.4f} | {rel} | {r['e2e_median']:.2f} | {r['h2d_fetches']} | {r['h2d_gib']:.0f} |")
for b in (16, 64):
    base = find(30, b, 'full')
    for w in (32, 128, 512):
        r = find(30, b, 'full', w)
        if r is None: continue
        rel = f"{100*(r['tpot_median']/base['tpot_median']-1):+.1f}%" if base else '-'
        md.append(f"| C30 | B{b} | Full Gate (W sweep) | {w} | {r['tpot_median']:.4f} | {r['tpot_min']:.4f}-{r['tpot_max']:.4f} | {rel} | {r['e2e_median']:.2f} | {r['h2d_fetches']} | {r['h2d_gib']:.0f} |")
(OUT / 'TPOT_TABLE.md').write_text('\n'.join(md) + '\n')
# figure: TPOT by policy, C30/C60
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'axes.axisbelow': True})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
for ax, p in zip(axes, (30, 60)):
    x = np.arange(3); w = .2
    for k, s in enumerate(('lru', 'count', 'selected', 'full')):
        vals = [find(p, b, s) for b in (8, 16, 64)]
        h = [v['tpot_median'] if v else np.nan for v in vals]
        lo = [v['tpot_median'] - v['tpot_min'] if v else 0 for v in vals]; hi = [v['tpot_max'] - v['tpot_median'] if v else 0 for v in vals]
        ax.bar(x + (k - 1.5) * w, h, w * .9, yerr=[lo, hi], capsize=1.5, color=COL[s], label=NAME[s], edgecolor='white', linewidth=.5,
               error_kw=dict(lw=.6))
    ax.set_xticks(x, ['B8', 'B16', 'B64']); ax.set_ylabel('TPOT (s, median of 3)'); ax.set_title(f'C{p} expert cache (live, L512/O64)', fontsize=8)
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.06), frameon=False, fontsize=7)
fig.tight_layout()
for ext in ('pdf', 'png'): fig.savefig(OUT / f'FIG_C1_TPOT.{ext}', dpi=200, bbox_inches='tight')
print('\n'.join(md))
