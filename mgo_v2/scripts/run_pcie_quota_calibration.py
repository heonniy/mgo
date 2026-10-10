"""Calibrate short Qwen-expert H2D bursts on every eight-GPU subset."""

import itertools
import json
import random
import time
from pathlib import Path

import run_full_pinned_r4 as owner
import run_pcie_rank_concurrency as bench


ROOT = Path('/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010')
GPUS = tuple(range(8))
COPIES = (1, 2, 4, 8, 32)


def main():
    assert owner.host_available() >= 384 * 2**30
    assert not (ROOT / 'STOP').exists()
    ROOT.mkdir(parents=True, exist_ok=True)
    groups = [group for n in range(1, 9)
              for group in itertools.combinations(GPUS, n)]
    assert len(groups) == 255
    random.Random(20261010).shuffle(groups)
    bench.GPUS = GPUS
    bench.SIZES = {'Qwen expert': 9 * bench.MIB}
    state = dict(status='RUNNING', gpus=list(GPUS), subsets=255,
                 copies=list(COPIES), repeats=2, started=time.time())
    bench.write(ROOT / 'STATUS.json', state)
    stopped = []
    try:
        stopped = owner.stop_target_idle()
        deadline = time.monotonic() + 40
        while bench.gpu_processes() and time.monotonic() < deadline:
            time.sleep(1)
        assert not bench.gpu_processes(), 'GPU occupied after owned-load stop'
        assert min(bench.gpu_free().values()) >= 2048
        state['stopped_owner_loads'] = stopped
        bench.write(ROOT / 'STATUS.json', state)
        for copies in COPIES:
            if (ROOT / 'STOP').exists():
                raise RuntimeError('owner STOP')
            # For n<8 misses every active rank gets one copy, so all subsets
            # matter. At n>=8 every rank is active; only the full-eight burst
            # is needed for longer per-rank queues.
            target_groups = groups if copies == 1 else [GPUS]
            rows = bench.run(target_groups, copies, 2, ROOT / f'RAW_c{copies}.json')
            assert len(rows) == 2 * len(target_groups)
            assert all(row['payload'] == 'Qwen expert' for row in rows)
            state[f'copies_{copies}'] = 'PASS'
            bench.write(ROOT / 'STATUS.json', state)
        state['status'] = 'PASS'
    except BaseException as error:
        state.update(status='FAIL', error=repr(error))
        raise
    finally:
        state['finished'] = time.time()
        state['restored_owner_loads'] = owner.restore_target_idle(stopped)
        bench.write(ROOT / 'STATUS.json', state)


if __name__ == '__main__':
    main()
