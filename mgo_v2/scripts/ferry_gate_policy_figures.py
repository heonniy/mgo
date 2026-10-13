"""Figures/tables for ferry_gate_policy_analysis outputs (Appendix C)."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = Path(sys.argv[1]); H = sys.argv[2] if len(sys.argv) > 2 else '256'
rows = []
for f in sorted(D.glob('*/POLICY_ROWS.csv')) if not (D / 'POLICY_ROWS.csv').exists() else [D / 'POLICY_ROWS.csv']:
    rows += list(csv.DictReader(open(f)))
tb = D.parent / (D.name + '_tiebreak') / 'POLICY_ROWS.csv'
if tb.exists(): rows += list(csv.DictReader(open(tb)))
B = ('8', '16', '64'); WS = (8, 16, 32, 64, 128, 256, 512, 1024)
COL = {'full-gate': '#2a78d6', 'selected-gate': '#eb6834', 'sel-count': '#1baf7a',
       'LFU-cumulative': '#eda100', 'LRU': '#e87ba4', 'count+gate-tiebreak': '#4a3aa7', 'OPT-BR (oracle)': '#333333'}
NAME = {'full-gate': 'Full Gate Score (FERRY)', 'selected-gate': 'Selected-only Gate', 'sel-count': 'Selection Count',
        'LFU-cumulative': 'LFU (cumulative)', 'LRU': 'LRU', 'count+gate-tiebreak': 'Sel. Count + gate tie-break',
        'OPT-BR (oracle)': 'Belady (oracle)'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': .6, 'axes.axisbelow': True})


def get(b, p, v, c, h=H):
    for r in rows:
        if r['local_batch'] == b and r['policy'] == p and r['variant'] == v and r['capacity_pct'] == c and r['horizon'] == h:
            return r
    return None


def save(fig, name):
    for ext in ('pdf', 'png'): fig.savefig(D / f'{name}_H{H}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---- Fig C.1: policy comparison (C30, C60) x (distinct, token-weighted)
POL = [('LRU', 'n/a'), ('LFU-cumulative', 'n/a'), ('sel-count', 'W128'), ('selected-gate', 'W128'), ('full-gate', 'W128')]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
for i, c in enumerate(('30', '60')):
    for j, (metric, lab) in enumerate((('hit_rate', 'Decode hit rate (%)'), ('token_hit_rate', 'Token-weighted hit rate (%)'))):
        ax = axes[i, j]; x = np.arange(len(B)); w = .15
        for k, (p, v) in enumerate(POL):
            vals = [100 * float(get(b, p, v, c)[metric]) for b in B]
            ax.bar(x + (k - 2) * w, vals, w * .9, color=COL[p], label=NAME[p], edgecolor='white', linewidth=.5)
            for xx, vv in zip(x + (k - 2) * w, vals):
                if vv < .5: ax.text(xx, .5, '0', ha='center', va='bottom', fontsize=5.5, color='#555')
        if metric == 'hit_rate':
            opt = [100 * float(get(b, 'OPT-BR (oracle)', 'n/a', c)[metric]) for b in B]
            for xx, o in zip(x, opt): ax.hlines(o, xx - 2.6 * w, xx + 2.6 * w, color='#333', lw=1, ls='--')
        ax.set_xticks(x, [f'B{b}' for b in B]); ax.set_ylabel(lab); ax.set_title(f'C{c} expert cache', fontsize=8)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color='#333', ls='--', lw=1)); labels.append('Belady oracle (distinct)')
fig.legend(handles, labels, ncol=6, loc='upper center', bbox_to_anchor=(.5, 1.04), frameon=False, fontsize=7)
fig.tight_layout(); save(fig, 'FIG_C1_POLICY_COMPARISON')

# ---- Fig C.2: why (ties, recovery, victim reuse)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
ax = axes[0]
for p in ('sel-count', 'selected-gate', 'full-gate'):
    ax.plot(WS, [100 * float(get('16', p, f'W{W}', '30')['tie_gt1_frac']) for W in WS], marker='o', ms=3, lw=1.5, color=COL[p], label=NAME[p])
ax.set_xscale('log', base=2); ax.set_xlabel('History window W (tokens)'); ax.set_ylabel('Evictions with tied min score (%)')
ax.set_title('(a) Score ties, B16 C30', fontsize=8); ax.legend(frameon=False, fontsize=6)
ax = axes[1]
for p in ('sel-count', 'count+gate-tiebreak', 'full-gate'):
    if get('16', p, 'W128', '30') is None: continue
    ax.plot(WS, [100 * float(get('16', p, f'W{W}', '30')['hit_rate']) for W in WS], marker='o', ms=3, lw=1.5, color=COL[p], label=NAME[p])
ax.set_xscale('log', base=2); ax.set_xlabel('History window W (tokens)'); ax.set_ylabel('Decode hit rate (%)')
ax.set_title('(b) Ties explain the count deficit', fontsize=8); ax.legend(frameon=False, fontsize=6)
ax = axes[2]
CAPS = ('20', '30', '40', '60'); MK = {'8': 'o', '16': 's', '64': '^'}
for p in ('selected-gate', 'sel-count'):
    for b in B:
        d = [100 * (float(get(b, p, 'W128', c)['hit_rate']) - float(get(b, 'full-gate', 'W128', c)['hit_rate'])) for c in CAPS]
        ax.plot([int(c) for c in CAPS], d, marker=MK[b], ms=3, lw=1.3, color=COL[p], label=f'{NAME[p]} B{b}')
ax.axhline(0, color='#2a78d6', lw=1.2); ax.set_xticks([20, 30, 40, 60], ['C20', 'C30', 'C40', 'C60'])
ax.set_ylabel('Hit rate - Full Gate (pp)'); ax.set_xlabel('Expert cache budget')
ax.set_title('(c) Gap to Full Gate vs budget (W=128)', fontsize=8); ax.legend(frameon=False, fontsize=5.5, ncol=1, loc='upper left')
fig.tight_layout(); save(fig, 'FIG_C2_WHY_GATE')

# ---- Fig W: window sweep, normalized to global batch
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=False)
GB = {'8': 32, '16': 64, '64': 256}; MK = {'8': 'o', '16': 's', '64': '^'}
for ax, c in zip(axes, ('30', '60')):
    for b in B:
        for p, ls in (('full-gate', '-'), ('sel-count', ':')):
            ys = np.array([100 * float(get(b, p, f'W{W}', c)['hit_rate']) for W in WS])
            ax.plot(np.array(WS) / GB[b], ys - ys.max(), ls=ls, marker=MK[b], ms=3, lw=1.3, color=COL[p],
                    label=f'{NAME[p]} B{b}')
    ax.axvline(1, color='#999', lw=.8); ax.set_xscale('log', base=2); ax.set_ylim(-6, .5)
    ax.set_xlabel('W / global decode batch (tokens per step)'); ax.set_ylabel('Hit rate - best W (pp)')
    ax.set_title(f'C{c}', fontsize=8)
axes[1].legend(frameon=False, fontsize=6, ncol=2)
fig.tight_layout(); save(fig, 'FIG_W_SWEEP')

# ---- Tables (markdown)
out = []
for c in ('30', '60'):
    out.append(f'\n### Policy comparison, C{c}, horizon {H} decode steps (W=128)\n')
    out.append('| Policy | ' + ' | '.join(f'B{b} hit % | B{b} tok % | B{b} H2D fetches | B{b} re-fetch' for b in B) + ' |')
    out.append('|---|' + '---:|' * 12)
    for p, v in POL[::-1] + [('random', 'n/a'), ('count+gate-tiebreak', 'W128'), ('OPT-BR (oracle)', 'n/a')]:
        if get('8', p, v, c) is None: continue
        cells = []
        for b in B:
            r = get(b, p, v, c)
            cells += [f"{100*float(r['hit_rate']):.2f}", f"{100*float(r['token_hit_rate']):.1f}", r['h2d_fetches'], r['refetch']]
        out.append(f'| {NAME.get(p, p)} | ' + ' | '.join(cells) + ' |')
    out.append(f'\n### Window sweep, C{c} (distinct hit % / token-weighted %)\n')
    out.append('| Policy | B | ' + ' | '.join(f'W={W}' for W in WS) + ' | all-history |')
    out.append('|---|---|' + '---:|' * (len(WS) + 1))
    for p in ('full-gate', 'selected-gate', 'sel-count'):
        for b in B:
            cells = [f"{100*float(get(b,p,f'W{W}',c)['hit_rate']):.2f} / {100*float(get(b,p,f'W{W}',c)['token_hit_rate']):.1f}" for W in WS]
            a = get(b, p, 'all', c)
            cells.append(f"{100*float(a['hit_rate']):.2f} / {100*float(a['token_hit_rate']):.1f}" if a else '-')
            out.append(f'| {NAME[p]} | B{b} | ' + ' | '.join(cells) + ' |')
out.append('\n### Prediction of future demand (W=128 scores; same at every capacity)\n')
out.append('| Score | B | Spearman h=1 | h=4 | h=16 | AUC next-step active | AUC among unselected-in-window | unselected share of next-step actives |')
out.append('|---|---|---:|---:|---:|---:|---:|---:|')
for p, v in (('full-gate', 'W128'), ('selected-gate', 'W128'), ('sel-count', 'W128'), ('LFU-cumulative', 'n/a'), ('LRU', 'n/a')):
    for b in B:
        r = get(b, p, v, '30')
        out.append(f"| {NAME[p]} | B{b} | " + ' | '.join(f"{float(r[k]):.3f}" for k in ('spearman_h1', 'spearman_h4', 'spearman_h16', 'auc_next_active', 'auc_next_active_unselected', 'unselected_share_of_next_active')) + ' |')
(D / f'TABLES_H{H}.md').write_text('\n'.join(out) + '\n')
print('wrote figures and', D / f'TABLES_H{H}.md')
