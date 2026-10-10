"""Guarded R8 matched-token comparison of grouped quota policies."""

import json
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
GUARD = PKG / 'scripts/run_qwen_r8_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TABLE = HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json'
TEACHER = HERE.parent / 'pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json'
POLICIES = (('group_near', 'LA_CA_NEAR'),
            ('group_fast', 'NEAR_FAST'),
            ('group_pcie', 'NEAR_PCIE'))


def write(data):
    path = HERE / 'STATUS.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(label, policy, attempt, repeats, smoke=False):
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
    if smoke:
        command += ['--smoke']
    else:
        command += ['--teacher-tokens', str(TEACHER)]
    if policy != 'LA_CA_NEAR':
        command += ['--quota-table', str(TABLE)]
    subprocess.run(command, check=True, cwd=PKG.parent)
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    assert json.loads(TABLE.read_text())['status'] == 'PASS'
    assert json.loads(TEACHER.read_text())['status'] == 'FROZEN'
    status = {'status': 'RUNNING', 'started': time.time()}
    write(status)
    try:
        for label, policy in POLICIES:
            assert not (HERE / 'STOP').exists()
            run(label, policy, 1, 1, smoke=True)
            first = run(label, policy, 1, 2)
            rows = [json.loads((first / f'repeat{i}.json').read_text())
                    for i in (1, 2)]
            gaps = {metric: 100 * abs(rows[0][metric] - rows[1][metric]) /
                    statistics.mean(x[metric] for x in rows)
                    for metric in ('TPOT', 'E2E')}
            status[label] = {'first_pair_gap_pct': gaps,
                             'third_repeat': max(gaps.values()) > 2}
            if status[label]['third_repeat']:
                run(label, policy, 2, 1)
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
