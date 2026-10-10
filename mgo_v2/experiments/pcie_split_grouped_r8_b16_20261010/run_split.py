"""Interleaved guarded R8 grouped comparison: NEAR, FAST, NEAR_SPLIT."""

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
TABLES = {'LA_CA_NEAR': None,
          'NEAR_FAST': HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json',
          'NEAR_SPLIT': HERE.parent / 'pcie_haq_replay_20261010/SPLIT_LOOKUP.json'}
LABELS = {'LA_CA_NEAR': 'split_near', 'NEAR_FAST': 'split_fast', 'NEAR_SPLIT': 'split_split'}
ORDERS = (('LA_CA_NEAR', 'NEAR_FAST', 'NEAR_SPLIT'),
          ('NEAR_FAST', 'NEAR_SPLIT', 'LA_CA_NEAR'),
          ('NEAR_SPLIT', 'LA_CA_NEAR', 'NEAR_FAST'))
REPEATS = 2


def write(data):
    path = HERE / 'STATUS.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(policy, attempt, repeats, smoke=False):
    label = LABELS[policy]
    kind = 'smoke' if smoke else 'full'
    output = ROOT / 'jobs' / f'{label}_{kind}_v{attempt}'
    if output.exists():
        status = json.loads((output / 'status.json').read_text())
        assert status['status'] == 'PASS' and status['repeats'] == repeats
        return output
    command = [PYTHON, '-u', str(GUARD), '--system', 'ours',
               '--ours-mode', 'N', '--job-label', label,
               '--ours-policy', policy, '--attempt', str(attempt),
               '--repeats', str(repeats), '--quiet-2367']
    command += ['--smoke'] if smoke else ['--teacher-tokens', str(TEACHER)]
    if TABLES[policy] is not None:
        command += ['--quota-table', str(TABLES[policy])]
    subprocess.run(command, check=True, cwd=PKG.parent)
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status)
    try:
        run('NEAR_SPLIT', 1, 1, smoke=True)
        for attempt, order in enumerate(ORDERS, 1):
            for policy in order:
                assert not (HERE / 'STOP').exists()
                out = run(policy, attempt, REPEATS)
                status['jobs'].append({'policy': policy, 'attempt': attempt, 'output': str(out)})
                write(status)
        status['status'] = 'PASS'
    except BaseException as error:
        status.update(status='FAIL', error=repr(error))
        raise
    finally:
        status['finished'] = time.time()
        write(status)


if __name__ == '__main__':
    main()
