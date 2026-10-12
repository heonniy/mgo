"""Microbench-optimal quota tables per miss count m (0..128), per payload.

MB_BAL: best balanced quota (max-min <= 1): every choice of which ranks take the m % 8 extra copies.
MB_OPT: unconstrained best quota: start from MB_BAL, then single-copy moves while the makespan drops
        (ties keep the more balanced row). Rows with max-min > 1 are reported.
Also reports the model makespan of NEAR (extras to ranks 0,1,..) and FAST (extras 7,6,5,4,..).
"""
import itertools
import json
from pathlib import Path

from fluid import W, load_rates, makespan

HERE = Path(__file__).resolve().parent
PAYLOADS = {'qwen': 'Qwen expert', 'deepseek': 'DeepSeek expert'}
FAST_ORDER = [7, 6, 5, 4, 3, 2, 1, 0]  # pcie_quota_r8_b16_20261010/LOOKUP.json NEAR_FAST


def balanced(m, extra):
    q = [m // W] * W
    for r in extra:
        q[r] += 1
    return q


def best_balanced(m, rates):
    rem = m % W
    cands = [balanced(m, c) for c in itertools.combinations(range(W), rem)]
    return min(cands, key=lambda q: (round(makespan(q, rates), 9), q))


def improve(q, rates):
    q = list(q); cur = makespan(q, rates)
    while True:
        best = None
        for i in range(W):
            if q[i] == 0:
                continue
            for j in range(W):
                if i == j:
                    continue
                t = q[:]; t[i] -= 1; t[j] += 1
                c = makespan(t, rates)
                if c < cur - 1e-9 and (best is None or c < best[0] - 1e-12):
                    best = (c, t)
        if best is None:
            return q, cur
        cur, q = best


def main():
    report = {}
    for model, payload in PAYLOADS.items():
        rates = load_rates(payload)
        bal, opt, rows = [], [], []
        for m in range(129):
            b = best_balanced(m, rates); o, to = improve(b, rates)
            bal.append(b); opt.append(o)
            near = balanced(m, range(m % W)); fast = balanced(m, FAST_ORDER[:m % W])
            rows.append(dict(m=m, near=makespan(near, rates), fast=makespan(fast, rates),
                             mb_bal=makespan(b, rates), mb_opt=to, mb_bal_q=b, mb_opt_q=o,
                             opt_spread=max(o) - min(o)))
        for key, lut in (('MB_BAL', bal), ('MB_OPT', opt)):
            (HERE / f'{key}_{model}.json').write_text(json.dumps(dict(
                status='PASS', physical_gpus=list(range(8)), payload=payload, rule=__doc__.strip().splitlines()[0] + f' [{key}]',
                quota_lut={'NEAR_SPLIT': lut, 'NEAR_FAST': lut}), separators=(',', ':')) + '\n')
        report[model] = rows
    (HERE / 'TABLES_REPORT.json').write_text(json.dumps(report, indent=0) + '\n')
    for model, rows in report.items():
        unb = [r['m'] for r in rows if r['opt_spread'] > 1]
        print(model, 'MB_OPT rows with max-min>1:', len(unb), 'of 129; first', unb[:8])
        for m in (12, 20, 40, 44, 48, 56, 64, 68, 72):
            r = rows[m]
            print(f"  m={m:3d} NEAR {r['near']:.4f} FAST {r['fast']:.4f} MB_BAL {r['mb_bal']:.4f} {r['mb_bal_q']}  MB_OPT {r['mb_opt']:.4f} {r['mb_opt_q']}")


if __name__ == '__main__':
    main()
