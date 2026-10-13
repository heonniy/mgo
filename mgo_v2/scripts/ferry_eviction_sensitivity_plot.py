"""Plot FIG_EVICTION_SENSITIVITY from EVICTION_SENSITIVITY.csv (CPU only)."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'experiments/ferry_appendix_sensitivity_20261011/eviction'
rows = [r for r in csv.DictReader((OUT / 'EVICTION_SENSITIVITY.csv').open()) if r['feasible'] == 'True']
PANELS = [('Qwen3-30B-A3B', '8'), ('Qwen3-30B-A3B', '16'), ('Qwen3-30B-A3B', '64'), ('DeepSeek-V2-Lite', '16')]
STYLE = [  # policy, label, color, marker, linestyle
    ('gate-score', 'Gate-score (W128)', '#2a78d6', 'o', '-'),
    ('LFU-cumulative', 'LFU-cumulative', '#eb6834', 's', '-'),
    ('LFU-reset', 'LFU-reset', '#1baf7a', '^', '-'),
    ('LRU-cumulative', 'LRU (reset = cumulative)', '#eda100', 'D', '-'),
    ('random', 'Random (seed 1234)', '#e87ba4', 'v', '-'),
    ('OPT-BR (oracle)', 'Belady OPT under BR (oracle)', '#555555', None, '--'),
]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
for col, (model, batch) in enumerate(PANELS):
    for row, metric in enumerate(('hit_rate', 'token_weighted_hit_rate')):
        ax = axes[row, col]
        for pol, label, color, marker, ls in STYLE:
            pts = sorted((float(r['capacity_frac']) * 100, float(r[metric]) * 100) for r in rows
                         if r['model'] == model and r['batch'] == batch and r['policy'] == pol)
            ax.plot(*zip(*pts), ls=ls, color=color, lw=1.8 if marker else 1.4, marker=marker, ms=3.5, label=label)
        ax.axvline(30, color='#999999', lw=0.8, ls=':')
        ax.grid(alpha=0.25, lw=0.5)
        if row == 0:
            ax.set_title(f'{model.split("-")[0]} {"" if model.startswith("Qwen") else "V2-Lite "}B{batch}/rank'
                         + ('' if model.startswith('Qwen') else ' (63 steps)'))
            ax.text(31, 92, 'C30', color='#777777', fontsize=8)
        if col == 0:
            ax.set_ylabel(('Distinct-expert' if row == 0 else 'Token-weighted') + ' decode hit rate (%)')
        if row == 1:
            ax.set_xlabel('MAIN cache budget (% of all experts)')
axes[0, 0].set_ylim(-2, 102); axes[0, 0].set_xlim(8, 102)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'FIG_EVICTION_SENSITIVITY.{ext}', dpi=200, bbox_inches='tight')
