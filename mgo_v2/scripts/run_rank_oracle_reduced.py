#!/usr/bin/env python3
"""Owner-authorized reduced matrix; reuse the frozen worker and memory guards."""
import fcntl
import json

from run_rank_oracle_study import ROOT, OUT, cell, run, trace


def main():
    amendment = json.loads((OUT / 'scope_amendment.json').read_text())
    lock = (ROOT / 'study.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert json.loads((ROOT / 'timing_b4/status.json').read_text())['status'] == 'PASS'
    for batch in (8, 16):
        plan = f'plan_b{batch}'
        assert json.loads((ROOT / plan / 'status.json').read_text())['status'] == 'PASS'
        cells = []
        for repeat, order in enumerate(amendment['amended_timing'][f'batch{batch}_orders']):
            cells.extend(cell(batch, p, f'b{batch}_{p}_rep{repeat}', 'replay', trace(plan, plan)) for p in order)
        run(f'timing_b{batch}', cells)


if __name__ == '__main__':
    main()
