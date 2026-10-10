"""Repeat only noisy short-burst subsets and all-eight burst anchors."""

import json
import time
from pathlib import Path

import run_full_pinned_r4 as owner
import run_pcie_rank_concurrency as bench


ROOT = Path('/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010')


def main():
    assert json.loads((ROOT / 'STATUS.json').read_text())['status'] == 'PASS'
    bench.GPUS = tuple(range(8))
    bench.SIZES = {'Qwen expert': 9 * bench.MIB}
    base = json.loads((ROOT / 'RAW_c1.json').read_text())
    bad = {tuple(row['gpus']) for row in base if row['start_skew_ms'] > .2}
    assert len(bad) < 16
    groups = sorted(bad | {tuple(range(8))})
    state = dict(status='RUNNING', noisy_groups=[list(g) for g in sorted(bad)],
                 full8_repeats=16, started=time.time())
    bench.write(ROOT / 'REFINE_STATUS.json', state)
    stopped = []
    try:
        assert owner.host_available() >= 384 * 2**30
        stopped = owner.stop_target_idle()
        deadline = time.monotonic() + 40
        while bench.gpu_processes() and time.monotonic() < deadline:
            time.sleep(1)
        assert not bench.gpu_processes()
        assert min(bench.gpu_free().values()) >= 2048
        for copies in (1, 2, 4, 8, 32):
            selected = groups if copies == 1 else [tuple(range(8))]
            rows = bench.run(selected, copies, 16, ROOT / f'REFINE_c{copies}.json')
            assert len(rows) == 16 * len(selected)
            state[f'copies_{copies}'] = 'PASS'
            bench.write(ROOT / 'REFINE_STATUS.json', state)
        state['status'] = 'PASS'
    except BaseException as error:
        state.update(status='FAIL', error=repr(error))
        raise
    finally:
        state['finished'] = time.time()
        state['restored_owner_loads'] = owner.restore_target_idle(stopped)
        bench.write(ROOT / 'REFINE_STATUS.json', state)


if __name__ == '__main__':
    main()
