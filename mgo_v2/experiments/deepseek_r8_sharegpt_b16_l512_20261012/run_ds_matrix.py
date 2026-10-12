"""Interleaved DeepSeek R8 serial + inline H2D comparison of quota/placement baselines (teacher tokens)."""

import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
ROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012')
GUARD = PKG / 'scripts/run_deepseek_r8_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TEACHER = HERE / 'FROZEN_TOKENS.json'
TABLE = HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json'
ARMS = {'ds_near': 'LA_CA_NEAR', 'ds_fast': 'NEAR_FAST', 'ds_fastrand': 'FAST_RANDOM',
        'ds_rquota': 'RANDOM_QUOTA_NEAR'}
N = list(ARMS)
ORDERS = ((N[0], N[1], N[2], N[3]), (N[3], N[2], N[1], N[0]), (N[1], N[0], N[3], N[2]))


def write(data):
    path = HERE / 'STATUS.json'
    tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def run(label, attempt):
    policy = ARMS[label]
    output = ROOT / 'jobs' / f'{label}_full_v{attempt}'
    if output.exists():
        assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
        return output
    cmd = [PYTHON, '-u', str(GUARD), '--decode-policy', policy, '--repeats', '2', '--attempt', str(attempt),
           '--inline-demand-h2d', '--h2d-serial-ablation', '--teacher-tokens', str(TEACHER),
           '--job-label', label]
    if policy in ('NEAR_FAST', 'FAST_RANDOM'):
        cmd += ['--quota-table', str(TABLE)]
    subprocess.run(cmd, check=True, cwd=PKG.parent)
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status)
    try:
        for attempt, order in enumerate(ORDERS, 1):
            for label in order:
                assert not (HERE / 'STOP').exists()
                status['jobs'].append({'label': label, 'policy': ARMS[label], 'attempt': attempt,
                                       'output': str(run(label, attempt))})
                write(status)
        status['status'] = 'PASS'
    except BaseException as error:
        status.update(status='FAIL', error=repr(error)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
