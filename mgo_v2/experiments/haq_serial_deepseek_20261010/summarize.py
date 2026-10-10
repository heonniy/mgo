"""Rebuild SUMMARY.json and RESULTS tables from raw/ (one process per cell, decode policy per target)."""
import json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
POLICIES = ('BW', 'BR', 'STATIC_BLOCK', 'LA_CA_NEAR', 'HAQ')
out = dict(status='PASS', units='TPOT seconds/token', cells={})
for C in (50, 30):
    pol = {p: [] for p in POLICIES}
    for job in sorted((HERE / 'raw').glob(f'ds_*_c{C}_v1')):
        k = 1
        while (job / f'repeat{k}.json').exists():
            s = json.loads((job / f'repeat{k}.json').read_text())
            ranks = [json.loads((job / f'repeat{k}_rank{r}.json').read_text()) for r in range(4)]
            pol[s['decode_policy']].append(dict(job=job.name, target=k, TPOT=s['TPOT'], TTFT=s['TTFT'], E2E=s['E2E'],
                argmax_hash=ranks[0]['argmax_hash'], decode_fetches_per_rank=ranks[0]['decode_fetches_per_rank']))
            k += 1
    bw = st.median(x['TPOT'] for x in pol['BW'])
    cell = {}
    for p, rows in pol.items():
        if not rows: continue
        m = st.median(x['TPOT'] for x in rows)
        cell[p] = dict(targets=rows, n=len(rows), median=m, vs_BW_pct=(m / bw - 1) * 100, vs_BW_ms=(m - bw) * 1000,
                       hashes=sorted({x['argmax_hash'] for x in rows}))
    out['cells'][f'DeepSeekV2Lite_ShareGPT_R4_C{C}_B16_L512_O64'] = cell
(HERE / 'SUMMARY.json').write_text(json.dumps(out, indent=1) + '\n')
for name, cell in out['cells'].items():
    print(f'### {name}\n')
    print('| Policy | Targets (job: order → TPOT) | n | Median | vs BW | Decode copies by rank (first target) |')
    print('|---|---|---|---|---|---|')
    for p in POLICIES:
        if p not in cell: continue
        c = cell[p]
        t = ', '.join(f"{x['job'].replace('ds_','').replace('_v1','')}:{x['target']}→{x['TPOT']:.4f}" for x in c['targets'])
        print(f"| {p} | {t} | {c['n']} | {c['median']:.4f} | {c['vs_BW_pct']:+.2f}% ({c['vs_BW_ms']:+.1f} ms) | {c['targets'][0]['decode_fetches_per_rank']} |")
    print()
