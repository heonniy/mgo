"""Summarize the interleaved R8 grouped NEAR / FAST / NEAR_SPLIT comparison (NEAR = baseline)."""

import json
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
TEACHER = json.loads((HERE.parent / 'pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json').read_text())
POLICIES = (('LA_CA_NEAR', 'split_near', 'NEAR'), ('NEAR_FAST', 'split_fast', 'FAST'),
            ('NEAR_SPLIT', 'split_split', 'SPLIT'))
METRICS = ('TTFT', 'TPOT', 'E2E', 'throughput')


def load(policy, label):
    rows = []
    for attempt in (1, 2, 3):
        path = RAW / f'{label}_full_v{attempt}'
        if not (path / 'status.json').exists():
            continue
        status = json.loads((path / 'status.json').read_text())
        if status['status'] == 'RUNNING':
            continue  # partial summary while the matrix is still running
        assert status['status'] == 'PASS' and status['ours_mode'] == 'N'
        command = status['command']
        assert command[command.index('--policy') + 1] == policy
        for repeat in range(1, status['repeats'] + 1):
            sample = json.loads((path / f'repeat{repeat}.json').read_text())
            ranks = [json.loads((path / f'repeat{repeat}_rank{r}.json').read_text()) for r in range(8)]
            assert all(x['phase'] == 'target' and x['expert_cache_start'] == 'empty'
                       and x['policy'] == policy and x['forced_continuation']
                       and x['grouped_decode_mode'] == 'hit_then_miss'
                       and x['validation']['status'] == 'PASS'
                       and x['validation']['controller']['quota_violations'] == 0 for x in ranks)
            diff = sum(int(np.sum(np.asarray(x['tokens']) != np.asarray(TEACHER['rank_tokens'][str(r)])))
                       for r, x in enumerate(ranks))
            rows.append(dict(attempt=attempt, repeat=repeat, commit=status['source_commit'],
                             copies=[x['validation']['scheduler']['copies'] for x in ranks],
                             token_diff=diff, **{m: sample[m] for m in METRICS}))
    return rows


def main():
    data = {name: load(policy, label) for policy, label, name in POLICIES}
    base = {m: statistics.median(r[m] for r in data['NEAR']) for m in METRICS}
    lines = ['| Policy | n | TPOT median [range], s | vs NEAR | E2E median [range], s | vs NEAR | TTFT median, s | token diff vs teacher (median) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    summary = {}
    for name, rows in data.items():
        if not rows:
            continue
        med = {m: statistics.median(r[m] for r in rows) for m in METRICS}
        rng = {m: (min(r[m] for r in rows), max(r[m] for r in rows)) for m in METRICS}
        rel = {m: 100 * (med[m] / base[m] - 1) for m in METRICS}
        # Paired by round: each round runs all three policies back to back.
        paired = []
        for attempt in sorted({r['attempt'] for r in rows}):
            mine = [r['TPOT'] for r in rows if r['attempt'] == attempt]
            near = [r['TPOT'] for r in data['NEAR'] if r['attempt'] == attempt]
            if mine and near:
                paired.append(100 * (statistics.mean(mine) / statistics.mean(near) - 1))
        copies = np.median(np.array([r['copies'] for r in rows]), 0).astype(int).tolist()
        summary[name] = dict(n=len(rows), median=med, range=rng, vs_near_pct=rel,
                             paired_round_TPOT_vs_near_pct=paired, copies_by_rank_median=copies,
                             fetches_median=int(statistics.median(sum(r['copies']) for r in rows)),
                             samples=rows)
        lines.append(f"| {name} | {len(rows)} | {med['TPOT']:.4f} [{rng['TPOT'][0]:.4f}, {rng['TPOT'][1]:.4f}] | {rel['TPOT']:+.2f}% "
                     f"| {med['E2E']:.3f} [{rng['E2E'][0]:.3f}, {rng['E2E'][1]:.3f}] | {rel['E2E']:+.2f}% | {med['TTFT']:.3f} "
                     f"| {statistics.median(r['token_diff'] for r in rows):.0f} |")
    lines += ['', '| Policy | Paired per-round TPOT vs NEAR | Copies by rank (median) | Total fetches |', '|---|---|---|---:|']
    for name, s in summary.items():
        lines.append(f"| {name} | {', '.join(f'{x:+.2f}%' for x in s['paired_round_TPOT_vs_near_pct'])} | {s['copies_by_rank_median']} | {s['fetches_median']} |")
    (HERE / 'RESULTS.json').write_text(json.dumps(summary, indent=1) + '\n')
    (HERE / 'RESULTS_TABLE.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
