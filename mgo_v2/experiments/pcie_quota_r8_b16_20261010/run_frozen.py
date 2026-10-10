"""Matched-token continuation control for the three R8 quota policies."""

import json
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
GUARD = HERE.parents[1] / 'scripts/run_qwen_r8_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TABLE = HERE / 'LOOKUP.json'
TEACHER = HERE / 'FROZEN_TOKENS.json'
POLICIES = (('near_frozen','LA_CA_NEAR'),
            ('fast_frozen','NEAR_FAST'),
            ('pcie_frozen','NEAR_PCIE'))


def write(data):
    path = HERE / 'FROZEN_STATUS.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def run(label, policy, attempt, repeats):
    output = RAW / 'jobs' / f'{label}_full_v{attempt}'
    if output.exists():
        status = json.loads((output / 'status.json').read_text())
        if status['status'] != 'PASS':
            raise RuntimeError(f'preserved failed attempt: {output}')
        return output
    command = [PYTHON, '-u', str(GUARD), '--system', 'ours',
               '--job-label', label, '--ours-policy', policy,
               '--teacher-tokens', str(TEACHER), '--attempt', str(attempt),
               '--repeats', str(repeats), '--quiet-2367']
    if policy != 'LA_CA_NEAR':
        command += ['--quota-table', str(TABLE)]
    subprocess.run(command, check=True, cwd=HERE.parents[2])
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS',output.name,flush=True)
    return output


def main():
    assert json.loads(TABLE.read_text())['status']=='PASS'
    assert json.loads(TEACHER.read_text())['status']=='FROZEN'
    assert json.loads((HERE/'PHYSICAL_STATUS.json').read_text())['status']=='PASS'
    status = dict(status='RUNNING', started=time.time())
    write(status)
    try:
        for label, policy in POLICIES:
            if (HERE / 'STOP').exists():
                raise RuntimeError('owner STOP')
            first = run(label, policy, 1, 2)
            rows = [json.loads((first / f'repeat{i}.json').read_text()) for i in (1,2)]
            gaps = {metric:100*abs(rows[0][metric]-rows[1][metric])/
                    statistics.mean(x[metric] for x in rows)
                    for metric in ('TPOT','E2E')}
            status[label] = dict(first_pair_gap_pct=gaps)
            if max(gaps.values()) > 2:
                run(label, policy, 2, 1)
                status[label]['third_repeat']=True
            else:
                status[label]['third_repeat']=False
            write(status)
        status['status']='PASS'
    except BaseException as error:
        status.update(status='FAIL',error=repr(error))
        raise
    finally:
        status['finished']=time.time()
        write(status)


if __name__=='__main__':
    main()
