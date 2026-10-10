"""Interleaved R8 grouped comparison: NEAR, FAST, HAQ, HAQ_FAST (common H2D submission mode)."""

import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
GUARD = PKG / 'scripts/run_qwen_r8_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TEACHER = HERE.parent / 'pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json'
FAST_TABLE = HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json'
INLINE = os.environ.get('INLINE_DEMAND_H2D', '0') == '1'
TAG = 'hi' if INLINE else 'hb'
ARMS = {f'{TAG}_near': 'LA_CA_NEAR', f'{TAG}_fast': 'NEAR_FAST',
        f'{TAG}_haq': 'HAQ', f'{TAG}_haqfast': 'HAQ_FAST'}
NAMES = list(ARMS)
ORDERS = ((NAMES[0], NAMES[1], NAMES[2], NAMES[3]), (NAMES[3], NAMES[2], NAMES[1], NAMES[0]))


def write(data):
    path = HERE / 'STATUS.json'
    tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def run(label, attempt, repeats, smoke=False):
    policy = ARMS[label]
    output = ROOT / 'jobs' / f'{label}_{"smoke" if smoke else "full"}_v{attempt}'
    if output.exists():
        status = json.loads((output / 'status.json').read_text())
        assert status['status'] == 'PASS' and status['repeats'] == repeats
        return output
    command = [PYTHON, '-u', str(GUARD), '--system', 'ours', '--ours-mode', 'N',
               '--job-label', label, '--ours-policy', policy, '--attempt', str(attempt),
               '--repeats', str(repeats), '--quiet-2367']
    command += ['--smoke'] if smoke else ['--teacher-tokens', str(TEACHER)]
    if policy == 'NEAR_FAST':
        command += ['--quota-table', str(FAST_TABLE)]
    if INLINE:
        command += ['--inline-demand-h2d']
    subprocess.run(command, check=True, cwd=PKG.parent)
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    status = {'status': 'RUNNING', 'inline_demand_h2d': INLINE, 'started': time.time(), 'jobs': []}
    write(status)
    try:
        run(f'{TAG}_haqfast', 1, 1, smoke=True)
        run(f'{TAG}_haq', 1, 1, smoke=True)
        for attempt, order in enumerate(ORDERS, 1):
            for label in order:
                assert not (HERE / 'STOP').exists()
                status['jobs'].append({'label': label, 'policy': ARMS[label], 'attempt': attempt,
                                       'output': str(run(label, attempt, 2))})
                write(status)
        status['status'] = 'PASS'
    except BaseException as error:
        status.update(status='FAIL', error=repr(error)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
