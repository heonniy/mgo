"""Rebuild SUMMARY.json and the RESULTS.md tables from raw/ (2 rounds x 3 targets per policy)."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
POLICIES = ('BW', 'BR', 'LA_CA_NEAR', 'HAQ')
out = dict(status='PASS', units='TPOT seconds/token', cells={})
for C in (50, 30):
    cell = {}
    for pol in POLICIES:
        rounds = []
        for rnd in (1, 2):
            d = HERE / 'raw' / f'c{C}_{pol}_r{rnd}'
            reps = []
            for k in (1, 2, 3):
                summary = json.loads((d / f'repeat{k}.json').read_text())
                ranks = [json.loads((d / f'repeat{k}_rank{r}.json').read_text()) for r in range(4)]
                reps.append(dict(target=k, TPOT=summary['TPOT'], TTFT=summary['TTFT'], E2E=summary['E2E'],
                                 argmax_hash=ranks[0]['argmax_hash'],
                                 copies_per_token_by_rank=[x['native_executor_counts']['serial_waited_copies'] / 63 for x in ranks],
                                 experts_per_token_by_rank=[x['native_executor_counts']['groups'] / 63 for x in ranks]))
            rounds.append(dict(round=rnd, label=d.name, targets=reps, median_TPOT=st.median(r['TPOT'] for r in reps)))
        samples = [r['TPOT'] for rd in rounds for r in rd['targets']]
        cell[pol] = dict(rounds=rounds, samples=samples, median=st.median(samples), min=min(samples), max=max(samples),
                         hashes=sorted({r['argmax_hash'] for rd in rounds for r in rd['targets']}))
    bw = cell['BW']
    for pol in POLICIES:
        c = cell[pol]
        c['vs_BW_pct'] = (c['median'] / bw['median'] - 1) * 100
        c['vs_BW_ms'] = (c['median'] - bw['median']) * 1000
        c['round_vs_BW_pct'] = [(c['rounds'][i]['median_TPOT'] / bw['rounds'][i]['median_TPOT'] - 1) * 100 for i in range(2)]
    out['cells'][f'Qwen3_ShareGPT_R4_C{C}_B16_L512_O64'] = cell
(HERE / 'SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n')
for name, cell in out['cells'].items():
    print(f'### {name}\n')
    print('| Policy | R1 t1 / t2 / t3 | R2 t1 / t2 / t3 | Median (6) | vs BW | R1 / R2 vs BW |')
    print('|---|---|---|---|---|---|')
    for pol in POLICIES:
        c = cell[pol]
        r = [' / '.join(f"{t['TPOT']:.4f}" for t in rd['targets']) for rd in c['rounds']]
        print(f"| {pol} | {r[0]} | {r[1]} | {c['median']:.4f} | {c['vs_BW_pct']:+.2f}% ({c['vs_BW_ms']:+.1f} ms) | {c['round_vs_BW_pct'][0]:+.2f}% / {c['round_vs_BW_pct'][1]:+.2f}% |")
    print('\n| Policy | Copies/token by rank (R1 t2) | Experts/token by rank (R1 t2) |')
    print('|---|---|---|')
    for pol in POLICIES:
        t = cell[pol]['rounds'][0]['targets'][1]
        print(f"| {pol} | {[round(x,1) for x in t['copies_per_token_by_rank']]} | {[round(x,1) for x in t['experts_per_token_by_rank']]} |")
    print()
