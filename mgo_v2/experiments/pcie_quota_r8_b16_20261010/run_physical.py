"""Serial guarded R8/B16 main_OURS quota comparison with bounded repeats."""

import json
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
SCRIPT = HERE.parents[1] / 'scripts/run_qwen_r8_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TABLE = HERE / 'LOOKUP.json'
POLICIES = (
    ('near_quota', 'LA_CA_NEAR'),
    ('fast_quota', 'NEAR_FAST'),
    ('pcie_quota', 'NEAR_PCIE'),
)


def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def run(label, policy, kind, attempt, repeats):
    output = ROOT / 'jobs' / f'{label}_{kind}_v{attempt}'
    if output.exists():
        state = json.loads((output / 'status.json').read_text())
        if state['status'] != 'PASS':
            raise RuntimeError(f'preserved failed attempt: {output}')
        return output
    command = [PYTHON, '-u', str(SCRIPT), '--system', 'ours',
               '--job-label', label, '--ours-policy', policy,
               '--attempt', str(attempt), '--repeats', str(repeats),
               '--quiet-2367']
    if kind == 'smoke':
        command.append('--smoke')
    if policy != 'LA_CA_NEAR':
        command += ['--quota-table', str(TABLE)]
    subprocess.run(command, check=True, cwd=HERE.parents[2])
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', output.name, flush=True)
    return output


def main():
    table = json.loads(TABLE.read_text())
    assert table['status'] == 'PASS' and table['physical_gpus'] == list(range(8))
    state = dict(status='RUNNING', started=time.time(), policies=[p for _, p in POLICIES])
    write(HERE / 'PHYSICAL_STATUS.json', state)
    try:
        for label, policy in POLICIES[1:]:
            run(label, policy, 'smoke', 1, 1)
        for label, policy in POLICIES:
            if (HERE / 'STOP').exists():
                raise RuntimeError('owner STOP')
            first = run(label, policy, 'full', 1, 2)
            rows = [json.loads((first / f'repeat{i}.json').read_text())
                    for i in (1, 2)]
            gaps = {key: abs(rows[0][key] - rows[1][key]) /
                    statistics.mean(row[key] for row in rows) * 100
                    for key in ('TPOT', 'E2E')}
            state[label] = dict(first_pair_gap_pct=gaps)
            if max(gaps.values()) > 2:
                run(label, policy, 'full', 2, 1)
                state[label]['third_repeat'] = True
            else:
                state[label]['third_repeat'] = False
            write(HERE / 'PHYSICAL_STATUS.json', state)
        state['status'] = 'PASS'
    except BaseException as error:
        state.update(status='FAIL', error=repr(error))
        raise
    finally:
        state['finished'] = time.time()
        write(HERE / 'PHYSICAL_STATUS.json', state)


if __name__ == '__main__':
    main()
