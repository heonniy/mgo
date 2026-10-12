"""Figure A: (a) H2D straggler, NEAR vs FAST quota; (b) EP cost = rank imbalance + pure comm for stage-2 placements."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / 'SUMMARY.json').read_text())
MODELS = (('qwen', 'Qwen3-30B-A3B'), ('deepseek', 'DeepSeek-V2-Lite'))
fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
for col, (m, title) in enumerate(MODELS):
    a = ax[0, col]
    arms = (('near', 'NEAR quota'), ('fast', 'FAST quota'))
    vals = [S[f'{m}/{k}']['h2d_straggler'] for k, _ in arms]
    a.bar([n for _, n in arms], [v['mean'] for v in vals], yerr=[v['std'] for v in vals],
          color=['#9aa5b1', '#2f6fb0'], capsize=4, width=0.55)
    base = vals[0]['mean']
    for i, v in enumerate(vals):
        a.text(i, v['mean'] * 1.01, f"{v['mean']:.2f}s" + ('' if i == 0 else f"\n({100 * (v['mean'] / base - 1):+.1f}%)"),
               ha='center', va='bottom', fontsize=9)
    a.set_ylim(0, base * 1.25)
    a.set_title(f'(a) {title}: H2D straggler (sum over layers of max-rank wait)', fontsize=9.5)
    a.set_ylabel('seconds per decode run')
    b = ax[1, col]
    arms = (('worst', 'FAST+Worst'), ('rand', 'FAST+Random'), ('fast', 'FAST+Near'), ('haqf', 'HAQ_FAST'))
    imb = [S[f'{m}/{k}']['own_imbalance'] for k, _ in arms]
    com = [S[f'{m}/{k}']['comm_cost'] for k, _ in arms]
    ep = [S[f'{m}/{k}']['ep_cost'] for k, _ in arms]
    x = range(len(arms))
    b.bar(x, [c['mean'] for c in com], color='#c9d6e3', label='pure comm (a2a + gather)', width=0.6)
    b.bar(x, [i['mean'] for i in imb], bottom=[c['mean'] for c in com], color='#d9534f',
          label='rank imbalance (H2D wait + compute)', width=0.6)
    b.errorbar(x, [e['mean'] for e in ep], yerr=[e['std'] for e in ep], fmt='none', ecolor='k', capsize=4)
    ref = S[f'{m}/fast']['ep_cost']['mean']
    for i, (e, (k, _)) in enumerate(zip(ep, arms)):
        tp = 100 * (S[f'{m}/{k}']['TPOT']['mean'] / S[f'{m}/near']['TPOT']['mean'] - 1)
        b.text(i, e['mean'] * 1.01, f"{e['mean']:.2f}s\nTPOT {tp:+.1f}% vs NEAR", ha='center', va='bottom', fontsize=8)
    b.set_xticks(list(x), [n for _, n in arms], fontsize=9)
    b.set_ylim(0, max(e['mean'] for e in ep) * 1.22)
    b.set_title(f'(b) {title}: EP cost (stage-2 placement)', fontsize=9.5)
    b.set_ylabel('seconds per decode run')
    b.legend(fontsize=8, loc='upper right')
fig.suptitle('R8, B16, C30, serial GEMM + inline H2D, teacher tokens; mean ± std over 3 runs', fontsize=10)
fig.tight_layout()
fig.savefig(HERE / 'figure_a.png', dpi=150)
print(HERE / 'figure_a.png')
