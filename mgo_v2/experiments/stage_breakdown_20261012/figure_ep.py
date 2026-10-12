"""Two-column figure: (a) H2D imbalance NEAR vs FAST; (b) EP cost = compute imbalance + comm cost (stacked).

Serial default environment, R8, inline H2D; ms per decode step (decode sums / 63), mean +- std over 3 runs.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'SUMMARY.json').read_text())
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
ARM = {'near': '#9a9893', 'fast': '#2a78d6'}
COMP, COMM = '#4a3aa7', '#b9b2ee'  # stacked components: compute imbalance (dark), comm cost (light)
SHORT = {'near': 'NEAR', 'fast': 'FAST+Near', 'rand': 'FAST+Random', 'worst': 'FAST+Worst', 'haqf': 'HAQ_FAST'}
EP_ARMS = ['rand', 'fast']
MODELS = [('qwen', 'Qwen3'), ('deepseek', 'DeepSeek')]
STEPS = 63  # decode steps per run (both models); values are converted to ms per decode step
m = lambda mk, a, k: S[f'{mk}/{a}'][k]['mean'] * 1000 / STEPS
sd = lambda mk, a, k: S[f'{mk}/{a}'][k]['std'] * 1000 / STEPS

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK, 'font.family': 'DejaVu Sans'})
fig, axes = plt.subplots(2, 2, figsize=(10.5, 4.3), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.6]))
for row, (mk, name) in enumerate(MODELS):
    # (a) H2D imbalance
    ax = axes[row, 0]; ax.set_facecolor(SURF)
    arms = ['near', 'fast']; y = [1, 0]
    vals = [m(mk, a, 'h2d_imbalance') for a in arms]; errs = [sd(mk, a, 'h2d_imbalance') for a in arms]
    ax.barh(y, vals, height=0.6, color=[ARM[a] for a in arms], edgecolor=SURF, linewidth=2, zorder=2)
    ax.errorbar(vals, y, xerr=errs, fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    for yi, v, a in zip(y, vals, arms):
        rel = '' if a == 'near' else f'  ({100 * (v / vals[0] - 1):+.0f}%)'
        ax.text(v + errs[arms.index(a)] + max(vals) * 0.03, yi, f'{v:.1f}{rel}', va='center', fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([SHORT[a] for a in arms]); ax.set_xlim(0, max(vals) * 1.55)
    # (b) EP cost stacked
    bx = axes[row, 1]; bx.set_facecolor(SURF)
    y = list(range(len(EP_ARMS)))[::-1]
    comp = [m(mk, a, 'compute_imbalance') for a in EP_ARMS]
    comm = [m(mk, a, 'comm_cost') for a in EP_ARMS]
    tot = [m(mk, a, 'ep_cost') for a in EP_ARMS]; terr = [sd(mk, a, 'ep_cost') for a in EP_ARMS]
    other = [t - c - k for t, c, k in zip(tot, comp, comm)]  # H2D-wait part of the own-imbalance term
    left_comm = [c + o for c, o in zip(comp, other)]
    bx.barh(y, [c + o for c, o in zip(comp, other)], height=0.62, color=COMP, edgecolor=SURF, linewidth=2, zorder=2)
    bx.barh(y, comm, left=left_comm, height=0.62, color=COMM, edgecolor=SURF, linewidth=2, zorder=2)
    bx.errorbar(tot, y, xerr=terr, fmt='none', ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
    base = tot[EP_ARMS.index('fast')]
    for yi, t, a in zip(y, tot, EP_ARMS):
        rel = '' if a == 'fast' else f'  ({100 * (t / base - 1):+.0f}%)'
        bx.text(t + terr[EP_ARMS.index(a)] + max(tot) * 0.02, yi, f'{t:.0f}{rel}', va='center', fontsize=8, color=INK)
    bx.set_yticks(y); bx.set_yticklabels([SHORT[a] for a in EP_ARMS]); bx.set_xlim(0, max(tot) * 1.28)
    for a_ in (ax, bx):
        a_.grid(axis='x', color=GRID, linewidth=0.8, zorder=0); a_.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            a_.spines[side].set_visible(False)
        a_.tick_params(axis='y', length=0)
    if row == 0:
        ax.set_title('(a) H2D: stage-1 quota', fontsize=10, color=INK, loc='left', fontweight='bold', pad=22)
        bx.set_title('(b) EP cost: stage-2 placement', fontsize=10, color=INK, loc='left', fontweight='bold', pad=22)
        bx.legend(handles=[Patch(color=COMP, label='load imbalance (compute + H2D wait)'),
                           Patch(color=COMM, label='communication cost')],
                  loc='lower left', bbox_to_anchor=(0.0, 1.07), ncol=2, frameon=False, fontsize=8,
                  handlelength=1.2, borderaxespad=0)
    else:
        ax.set_xlabel('H2D straggler imbalance (ms / decode step) ↓')
        bx.set_xlabel('EP cost = load imbalance + communication (ms / decode step) ↓')
    ax.annotate(name, xy=(-0.62, 0.5), xycoords='axes fraction', rotation=90, va='center', ha='center',
                fontsize=10, color=INK, fontweight='bold')
fig.suptitle('Stage-wise cost breakdown: R8, serial expert execution, inline H2D (per decode step, mean ± std, 3 runs)',
             x=0.01, ha='left', fontsize=10.5, color=INK)
fig.text(0.01, 0.005, '(a) relative to NEAR; (b) relative to FAST+Near (same FAST quota, stage-2 placement only).',
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0.04, 0.03, 1, 0.94))
fig.savefig(HERE / 'stage_breakdown_ep.png', dpi=200, facecolor=SURF)
fig.savefig(HERE / 'stage_breakdown_ep.pdf', facecolor=SURF)
print('saved')
