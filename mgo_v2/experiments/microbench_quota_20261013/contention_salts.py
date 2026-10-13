"""Worst/median balanced-random salts under induced contention (hogs on GPUs 0-3), Qwen.

Model: microbench subset rates with GPUs 0-3 slowed by the calibrated hog factor
(no-hog copy ratio 1.123 -> 1.288 at 1 stream, 1.724 at 2 streams; induced_contention_20261012/CALIBRATION*.json),
per-event miss counts from that level's NEAR rep-1 run.
"""
import json
from pathlib import Path

import fluid
from build_tables import FAST_ORDER, balanced
from seeds import imbalance, qwen_events, score

HERE = Path(__file__).resolve().parent
JOBS = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
LEVELS = {'l13': 1.123 / 1.288, 'l17': 1.123 / 1.724}


def main():
    base = fluid.load_rates('Qwen expert'); out = {}
    for level, f in LEVELS.items():
        rates = {m: [v * (f if r < 4 else 1.0) for r, v in enumerate(b)] for m, b in base.items()}
        ev = qwen_events(JOBS / f'icq{level}near_full_v1'); cache = {}
        sc = {s: score(ev, s, rates, cache) for s in range(1, 2001)}
        order = sorted(sc, key=lambda s: sc[s][0])
        ref = lambda g: sum(imbalance(g(m), rates, cache)[0] for _, m, _ in ev)
        out[level] = dict(slow_group_rate_factor=f, worst_salt=order[-1], worst=sc[order[-1]][0],
                          median_salt=order[len(order) // 2], median=sc[order[len(order) // 2]][0],
                          best_salt=order[0], best=sc[order[0]][0], salts_1_2_3=[sc[s][0] for s in (1, 2, 3)],
                          near=ref(lambda m: balanced(m, range(m % 8))), fast=ref(lambda m: balanced(m, FAST_ORDER[:m % 8])),
                          m_sequence_from=f'icq{level}near_full_v1')
        print(level, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[level].items()})
    (HERE / 'CONTENTION_SALTS.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
