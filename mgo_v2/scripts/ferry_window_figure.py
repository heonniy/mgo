"""Figure for the FERRY W-sensitivity appendix (reads ALL_ROWS.json)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path(sys.argv[1]); rows = json.loads((out / 'ALL_ROWS.json').read_text())['rows']
WS = [32, 64, 128, 256, 512, 1024]; BATCH = [8, 16, 64]
COL = {8: '#2a78d6', 16: '#eb6834', 64: '#1baf7a'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False})


def get(b, pct, variant, order='identity'):
    return next(r for r in rows if r['local_batch'] == b and r['capacity_pct'] == pct and r['variant'] == variant and r['order'] == order)


fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), constrained_layout=True)
for ax, pct in zip(axes[:3], (30, 40, 60)):
    for b in BATCH:
        y = [100 * get(b, pct, f'W{w}')['hit_rate'] for w in WS]
        ax.plot(WS, y, '-o', color=COL[b], lw=2, ms=4, label=f'local B{b} (global {4*b})')
        ax.axhline(100 * get(b, pct, 'all_history')['hit_rate'], color=COL[b], lw=1, ls='--')
        ax.axhline(100 * get(b, pct, 'current_event')['hit_rate'], color=COL[b], lw=1, ls=':')
    ax.axvline(128, color=MUTED, lw=1, alpha=.5)
    ax.set_xscale('log', base=2); ax.set_xticks(WS); ax.set_xticklabels(WS)
    ax.set_title(f'MAIN capacity {pct}%', color=INK); ax.set_xlabel('window W (tokens)')
    ax.grid(axis='y', color=GRID, lw=.6)
axes[0].set_ylabel('decode distinct-expert hit rate (%)')
ax = axes[3]
for b in BATCH:
    for h, ls in ((1, '-'), (16, '--')):
        ax.plot(WS, [get(b, 30, f'W{w}')[f'spearman_h{h}'] for w in WS], ls, marker='o', ms=3, lw=1.5, color=COL[b],
                label=f'B{b}, h={h}')
ax.axvline(128, color=MUTED, lw=1, alpha=.5)
ax.set_xscale('log', base=2); ax.set_xticks(WS); ax.set_xticklabels(WS); ax.grid(axis='y', color=GRID, lw=.6)
ax.set_title('score vs future demand', color=INK); ax.set_xlabel('window W (tokens)'); ax.set_ylabel('mean Spearman rho')
ax.legend(fontsize=6, frameon=False)
h, l = axes[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=MUTED, ls='--'), plt.Line2D([], [], color=MUTED, ls=':')]
l += ['all-history (cumulative mean)', 'current event only']
fig.legend(h, l, loc='outside lower center', ncol=5, fontsize=7, frameon=False)
for ext in ('pdf', 'png'): fig.savefig(out / f'FIG_WINDOW_SWEEP.{ext}', dpi=200)
