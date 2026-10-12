"""Figure: stage-wise cost breakdown (serial default environment), Qwen3 and DeepSeek R8, mean +- std over 3 reps."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'SUMMARY.json').read_text())
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COL = {'near': '#9a9893', 'fast': '#2a78d6', 'rand': '#eb6834', 'haqf': '#1baf7a', 'worst': '#eda100'}
LAB = {'near': 'NEAR (quota: remainder to GPU 0-3)', 'fast': 'FAST + Near placement',
       'rand': 'FAST + Random placement', 'worst': 'FAST + Worst placement', 'haqf': 'HAQ_FAST (hit-aware quota + Near)'}
SHORT = {'near': 'NEAR', 'fast': 'FAST+Near', 'rand': 'FAST+Random', 'worst': 'FAST+Worst', 'haqf': 'HAQ_FAST'}
PANELS = [('(a) H2D: quota', 'h2d_imbalance', ['near', 'fast'], 'H2D straggler imbalance (s) ↓'),
          ('(b) EP: compute', 'compute_imbalance', ['worst', 'rand', 'fast', 'haqf'], 'Expert-compute imbalance (s) ↓'),
          ('(c) EP: communication', 'comm_cost', ['worst', 'rand', 'fast', 'haqf'], 'All-to-all cost (s) ↓')]
MODELS = [('qwen', 'Qwen3-30B-A3B'), ('deepseek', 'DeepSeek-V2-Lite')]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(2, 3, figsize=(12, 4.6), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
for row, (mk, mname) in enumerate(MODELS):
    for col, (title, metric, arms, xlabel) in enumerate(PANELS):
        ax = axes[row, col]; ax.set_facecolor(SURF)
        vals = [S[f'{mk}/{a}'][metric]['mean'] for a in arms]
        errs = [S[f'{mk}/{a}'][metric]['std'] for a in arms]
        y = list(range(len(arms)))[::-1]
        ax.barh(y, vals, height=0.62, color=[COL[a] for a in arms], edgecolor=SURF, linewidth=2, zorder=2)
        ax.errorbar(vals, y, xerr=errs, fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        base = S[f'{mk}/fast'][metric]['mean'] if metric != 'h2d_imbalance' else S[f'{mk}/near'][metric]['mean']
        for yi, v, a in zip(y, vals, arms):
            rel = '' if (metric == 'h2d_imbalance' and a == 'near') or (metric != 'h2d_imbalance' and a == 'fast') \
                else f'  ({100 * (v / base - 1):+.0f}%)'
            ax.text(v + max(vals) * 0.03, yi, f'{v:.2f}{rel}', va='center', fontsize=8, color=INK)
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
                        ha='center', fontsize=10, color=INK, fontweight='bold')
fig.suptitle('Stage-wise cost breakdown: R8, serial expert execution, inline H2D (decode sums, mean ± std, 3 runs)',
             x=0.01, ha='left', fontsize=10.5, color=INK)
fig.text(0.01, 0.005, '(a) relative to NEAR; (b)(c) relative to FAST+Near. HAQ_FAST uses a hit-aware quota '
         '(H2D imbalance stays near FAST: Qwen 0.46 vs 0.44 s, DeepSeek 0.32 vs 0.27 s).',
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0.03, 0.03, 1, 0.94))
fig.savefig(HERE / 'stage_breakdown.png', dpi=200, facecolor=SURF)
fig.savefig(HERE / 'stage_breakdown.pdf', facecolor=SURF)
print('saved')
