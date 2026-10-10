"""Interleaved R8 grouped comparison of stage-2 placement under the FAST quota (inline H2D)."""

import json
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
ARMS = {'s2_near': 'LA_CA_NEAR', 's2_fastnear': 'NEAR_FAST',
        's2_fastrand': 'FAST_RANDOM', 's2_fastworst': 'FAST_WORST'}
N = list(ARMS)
ORDERS = ((N[0], N[1], N[2], N[3]), (N[3], N[2], N[1], N[0]))


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
               '--repeats', str(repeats), '--quiet-2367', '--inline-demand-h2d']
    command += ['--smoke'] if smoke else ['--teacher-tokens', str(TEACHER)]
    if policy != 'LA_CA_NEAR':
        command += ['--quota-table', str(FAST_TABLE)]
    subprocess.run(command, check=True, cwd=PKG.parent)
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    status = {'status': 'RUNNING', 'inline_demand_h2d': True, 'started': time.time(), 'jobs': []}
    write(status)
    try:
        run('s2_fastworst', 1, 1, smoke=True)
        run('s2_fastrand', 1, 1, smoke=True)
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
