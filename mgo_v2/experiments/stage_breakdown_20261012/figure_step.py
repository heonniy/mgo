"""Figure: stage-wise cost breakdown per decode step (ms), serial default environment, mean +- std over 3 reps.

Same data as figure.py (SUMMARY.json decode sums in s) divided by the 63 decode steps; EP panels keep
FAST+Random vs FAST+Near only. Also writes the plotted values to stage_breakdown_step.csv."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'SUMMARY.json').read_text())
STEPS = 63
MS = 1e3 / STEPS  # decode sum (s) -> ms per decode step
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'near': '#9a9893', 'fast': '#2a78d6', 'rand': '#eb6834', 'haqf': '#1baf7a', 'worst': '#eda100'}
LAB = {'near': 'NEAR (quota: remainder to GPU 0-3)', 'fast': 'FAST + Near placement',
       'rand': 'FAST + Random placement', 'worst': 'FAST + Worst placement', 'haqf': 'HAQ_FAST (hit-aware quota + Near)'}
SHORT = {'near': 'NEAR', 'fast': 'FAST+Near', 'rand': 'FAST+Random', 'worst': 'FAST+Worst', 'haqf': 'HAQ_FAST'}
PANELS = [('(a) H2D: quota', 'h2d_imbalance', ['near', 'fast'], 'H2D straggler imbalance (ms / decode step) ↓'),
          ('(b) EP: compute', 'compute_imbalance', ['rand', 'fast'], 'Expert-compute imbalance (ms / decode step) ↓'),
          ('(c) EP: communication', 'comm_cost', ['rand', 'fast'], 'All-to-all cost (ms / decode step) ↓')]
MODELS = [('qwen', 'Qwen3-30B-A3B'), ('deepseek', 'DeepSeek-V2-Lite')]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), facecolor=SURF)
rows_csv = []
for row, (mk, mname) in enumerate(MODELS):
    for col, (title, metric, arms, xlabel) in enumerate(PANELS):
        ax = axes[row, col]; ax.set_facecolor(SURF)
        vals = [S[f'{mk}/{a}'][metric]['mean'] * MS for a in arms]
        errs = [S[f'{mk}/{a}'][metric]['std'] * MS for a in arms]
        rows_csv += [(mk, title, metric, SHORT[a], v, e) for a, v, e in zip(arms, vals, errs)]
        y = list(range(len(arms)))[::-1]
        ax.barh(y, vals, height=0.62, color=[COL[a] for a in arms], edgecolor=SURF, linewidth=2, zorder=2)
        ax.errorbar(vals, y, xerr=errs, fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        base = (S[f'{mk}/fast'][metric]['mean'] if metric != 'h2d_imbalance' else S[f'{mk}/near'][metric]['mean']) * MS
        for yi, v, a in zip(y, vals, arms):
            rel = '' if (metric == 'h2d_imbalance' and a == 'near') or (metric != 'h2d_imbalance' and a == 'fast') \
                else f'  ({100 * (v / base - 1):+.0f}%)'
            ax.text(v + max(vals) * 0.03, yi, f'{v:.1f}{rel}', va='center', fontsize=8, color=INK)
        ax.set_yticks(y); ax.set_yticklabels([SHORT[a] for a in arms])
        ax.set_xlim(0, max(vals) * 1.45)
        ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0); ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis='y', length=0)
        if row == 0:
            ax.set_title(title, fontsize=10, color=INK, loc='left', fontweight='bold')
        if row == 1:
            ax.set_xlabel(xlabel)
        if col == 0:
            ax.annotate(mname, xy=(-0.55, 0.5), xycoords='axes fraction', rotation=90, va='center',
                        ha='center', fontsize=9.5, color=INK, fontweight='bold')
fig.suptitle('Stage-wise cost breakdown: R8, serial expert execution, inline H2D (ms per decode step, mean ± std, 3 runs)',
             x=0.01, ha='left', fontsize=10.5, color=INK)
fig.text(0.01, 0.005, '(a) relative to NEAR; (b)(c) relative to FAST+Near. Decode sums divided by 63 decode steps.',
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
fig.savefig(HERE / 'stage_breakdown_step.png', dpi=200, facecolor=SURF)
fig.savefig(HERE / 'stage_breakdown_step.pdf', facecolor=SURF)
import csv
with open(HERE / 'stage_breakdown_step.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['model', 'panel', 'metric', 'arm', 'ms_per_step_mean', 'ms_per_step_std']); w.writerows([(a, b, c, d, round(v, 3), round(e, 3)) for a, b, c, d, v, e in rows_csv])
print('saved')
