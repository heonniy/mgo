"""Figure: Qwen3 R8 H2D straggler imbalance (ms per decode step) per quota baseline at no contention, 1.3x and 1.7x.

Serial expert execution, inline H2D, Near placement, C30; mean +- std over 3 runs (SPLIT at no contention: 1 run,
1.44x burst table). No-contention NEAR/FAST/MB_BAL/BRAND_W/Random quota come from microbench_quota_20261013
(Random quota reused from stage_breakdown_20261012); the rest from this packet's SUMMARY.json.
Writes h2d_levels.png/pdf and h2d_levels.csv.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
IC = json.loads((HERE / 'SUMMARY.json').read_text())
MB = json.loads((HERE.parent / 'microbench_quota_20261013' / 'SUMMARY.json').read_text())
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ARMS = [('rq', 'Random quota', '#eb6834'), ('near', 'NEAR', '#9a9893'), ('bw', 'Balanced random (worst seed)', '#e87ba4'),
        ('brand', 'Balanced random (seed 1-3)', '#eda100'), ('fast', 'FAST', '#2a78d6'), ('mbbal', 'MB_BAL', '#1baf7a'),
        ('split', 'SPLIT', '#4a3aa7')]
MBKEY = {'near': 'NEAR', 'fast': 'FAST', 'mbbal': 'MB_BAL', 'bw': 'BRAND_W', 'rq': 'RANDOM_QUOTA'}
LEVELS = [('l00', 'No contention (~1.1x)'), ('l13', 'Induced 1.3x (GPUs 0-3)'), ('l17', 'Induced 1.7x (GPUs 0-3)')]


def cell(level, arm):
    if level == 'l00' and arm in MBKEY:
        v = MB[MBKEY[arm]]; return v['h2d_imbalance']['mean'], v['h2d_imbalance']['std'], v['n']
    v = IC.get(f'qwen/{level}/{arm}')
    return (v['h2d_imbalance']['mean'], v['h2d_imbalance']['std'], v['n']) if v else None


plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), facecolor=SURF, sharey=True)
rows = []
for ax, (level, title) in zip(axes, LEVELS):
    ax.set_facecolor(SURF)
    base = cell(level, 'near')[0]
    present = [(a, lab, c, cell(level, a)) for a, lab, c in ARMS]
    xmax = max(v[0] for *_, v in present if v)
    for i, (a, lab, c, v) in enumerate(present):
        y = len(ARMS) - 1 - i
        if v is None:
            ax.text(xmax * 0.02, y, 'not run', va='center', fontsize=8, color=INK2); continue
        m, s, n = v
        ax.barh(y, m, height=0.62, color=c, edgecolor=SURF, linewidth=2, zorder=2)
        ax.errorbar(m, y, xerr=s, fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        rel = '' if a == 'near' else f'  ({100 * (m / base - 1):+.0f}%)'
        ax.text(m + xmax * 0.03, y, f'{m:.1f}{rel}' + ('  n=1' if n == 1 else ''), va='center', fontsize=8, color=INK)
        rows.append((level, lab, round(m, 3), round(s, 3), n, round(100 * (m / base - 1), 2)))
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels([lab for _, lab, _ in ARMS][::-1])
    ax.set_xlim(0, xmax * 1.5); ax.set_title(title, fontsize=10, color=INK, loc='left', fontweight='bold')
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('H2D straggler imbalance (ms / decode step) ↓')
fig.suptitle('Qwen3-30B-A3B R8, C30: H2D imbalance by stage-1 quota under PCIe contention (per decode step, mean ± std, 3 runs)',
             x=0.01, ha='left', fontsize=10.5, color=INK)
fig.text(0.01, 0.01, 'Percent relative to NEAR at the same level. Near placement in all arms. SPLIT uses the level\'s table '
         '(no contention: 1.44x burst table, 1 run; 1.3x/1.7x: 1.29x/1.72x). Each panel has its own x-scale.',
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(HERE / 'h2d_levels.png', dpi=200, facecolor=SURF); fig.savefig(HERE / 'h2d_levels.pdf', facecolor=SURF)
with open(HERE / 'h2d_levels.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['level', 'quota', 'h2d_imbalance_ms_per_step_mean', 'std', 'n', 'vs_NEAR_pct']); w.writerows(rows)
print('saved')
