"""Summarize the DeepSeek R8 matrix from STATUS.json (NEAR = baseline)."""
import json, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
teacher = json.loads((HERE / 'FROZEN_TOKENS.json').read_text())
status = json.loads((HERE / 'STATUS.json').read_text())
rows, order = defaultdict(list), []
for job in status['jobs']:
    path = Path(job['output']); st = json.loads((path / 'status.json').read_text()); assert st['status'] == 'PASS'
    if job['label'] not in order:
        order.append(job['label'])
    for rep in (1, 2):
        s = json.loads((path / f'repeat{rep}.json').read_text())
        rk = [json.loads((path / f'repeat{rep}_rank{r}.json').read_text()) for r in range(8)]
        assert all(x['phase'] == 'target' and x['forced_continuation'] for x in rk)
        diff = sum(int((np.asarray(x['tokens']) != np.asarray(teacher['rank_tokens'][str(r)])).sum()) for r, x in enumerate(rk))
        rows[job['label']].append(dict(attempt=job['attempt'], rep=rep, TPOT=s['TPOT'], E2E=s['E2E'],
                                       fetches=s['decode_fetches_per_rank'],
                                       wait=[w / 1000 for w in s['serial_wait_ms_per_rank']], token_diff=diff))
base = statistics.median(r['TPOT'] for r in rows['ds_near'])
lines = ['| Arm | n | TPOT median [range], s | vs NEAR | per round vs NEAR | max H2D wait 0-3 / 4-7, s | decode fetches per rank (k) | token diff vs teacher |',
         '|---|---:|---:|---:|---|---|---|---:|']
for label in order:
    rs = rows[label]; tp = [r['TPOT'] for r in rs]; med = statistics.median(tp)
    per = []
    for a in sorted({r['attempt'] for r in rs}):
        mine = [r['TPOT'] for r in rs if r['attempt'] == a]; ref = [r['TPOT'] for r in rows['ds_near'] if r['attempt'] == a]
        if mine and ref: per.append(100 * (statistics.mean(mine) / statistics.mean(ref) - 1))
    w = np.median([r['wait'] for r in rs], 0); f = np.median([r['fetches'] for r in rs], 0)
    lines.append(f"| {label} | {len(rs)} | {med:.4f} [{min(tp):.4f}, {max(tp):.4f}] | {100 * (med / base - 1):+.2f}% | "
                 f"{', '.join(f'{x:+.2f}%' for x in per)} | {w[:4].max():.2f} / {w[4:].max():.2f} | "
                 f"{', '.join(f'{x / 1000:.1f}' for x in f)} | {statistics.median(r['token_diff'] for r in rs):.0f} |")
(HERE / 'RESULTS_TABLE.md').write_text('\n'.join(lines) + '\n')
(HERE / 'RESULTS.json').write_text(json.dumps({k: v for k, v in rows.items()}, indent=1) + '\n')
print('\n'.join(lines))
