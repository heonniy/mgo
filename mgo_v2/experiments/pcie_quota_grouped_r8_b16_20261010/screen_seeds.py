"""Screen frozen ShareGPT seeds and confirm the best observed PCIe gain."""

import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/pcie_quota_grouped_r8_b16_20261010/seeds')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
GUARD = HERE / 'seed_guard.py'
TABLE = HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json'
SEEDS = json.loads((HERE / 'SEEDS.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(data):
    target = HERE / 'SEED_STATUS.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(target)


def run(seed, label, policy, attempt, repeats=1, teacher=None):
    directory = RAW / f'seed{seed}'
    output = directory / 'jobs' / f'{label}_full_v{attempt}'
    if output.exists():
        status = json.loads((output / 'status.json').read_text())
        assert status['status'] == 'PASS' and status['repeats'] == repeats
        assert status['ours_mode'] == 'N' and status['workload_sha256'] == sha(directory / 'WORKLOADS.json')
        command = status['command']
        assert command[command.index('--policy') + 1] == policy
        assert command[command.index('--grouped-decode-mode') + 1] == 'hit_then_miss'
        assert '--compiled-dense' in command and '--prefetch-off' in command
        if teacher is not None:
            assert sha(Path(command[command.index('--teacher-tokens') + 1])) == sha(teacher)
        return output
    command = [PYTHON, '-u', str(GUARD), '--workloads', str(directory / 'WORKLOADS.json'),
               '--output-root', str(directory), '--system', 'ours', '--ours-mode', 'N',
               '--job-label', label, '--ours-policy', policy,
               '--attempt', str(attempt), '--repeats', str(repeats), '--quiet-2367']
    if teacher is not None:
        command += ['--teacher-tokens', str(teacher)]
    if policy != 'LA_CA_NEAR':
        command += ['--quota-table', str(TABLE)]
    subprocess.run(command, check=True, cwd=HERE.parents[2])
    assert json.loads((output / 'status.json').read_text())['status'] == 'PASS'
    print('PASS', seed, output.name, flush=True)
    return output


def freeze(seed, source):
    directory = RAW / f'seed{seed}'
    target = directory / 'TEACHER_TOKENS.json'
    status = json.loads((source / 'status.json').read_text())
    assert status['status'] == 'PASS' and status['physical_gpus'] == list(range(8))
    tokens = {}
    request_ids = {}
    for rank in range(8):
        row = json.loads((source / f'repeat1_rank{rank}.json').read_text())
        assert row['phase'] == 'target' and row['policy'] == 'LA_CA_NEAR'
        assert row['grouped_decode_mode'] == 'hit_then_miss'
        assert row['expert_cache_start'] == 'empty'
        assert len(row['tokens']) == len(row['request_ids']) == 16
        assert all(len(seq) == 64 for seq in row['tokens'])
        tokens[str(rank)] = row['tokens']
        request_ids[str(rank)] = row['request_ids']
    data = {'status': 'FROZEN',
            'cell': 'Qwen3_ShareGPT_R8_C30_B16_L512_O64',
            'local_batch': 16, 'output_tokens': 64,
            'source_job': str(source), 'source_commit': status['source_commit'],
            'workload_sha256': status['workload_sha256'],
            'rank_tokens': tokens, 'request_ids': request_ids}
    if target.exists():
        assert json.loads(target.read_text()) == data
    else:
        target.write_text(json.dumps(data, indent=2) + '\n')
    return target


def metric(job, key='TPOT', index=1):
    return float(json.loads((job / f'repeat{index}.json').read_text())[key])


def main():
    assert SEEDS['status'] == 'FROZEN' and SEEDS['seeds'] == [11, 23, 37, 53]
    assert json.loads(TABLE.read_text())['status'] == 'PASS'
    status = {'status': 'RUNNING', 'started': time.time(), 'seeds': SEEDS['seeds']}
    write(status)
    try:
        for index, row in enumerate(SEEDS['candidates']):
            seed = row['seed']
            assert not (HERE / 'STOP').exists()
            directory = RAW / f'seed{seed}'
            manifest = directory / 'WORKLOADS.json'
            assert str(manifest) == row['manifest'] and sha(manifest) == row['manifest_sha256']
            near = run(seed, 'seed_near', 'LA_CA_NEAR', 1)
            teacher = freeze(seed, near)
            order = [('seed_fast', 'NEAR_FAST'), ('seed_pcie', 'NEAR_PCIE')]
            if index % 2:
                order.reverse()
            jobs = {label: run(seed, label, policy, 1, teacher=teacher)
                    for label, policy in order}
            fast = metric(jobs['seed_fast'])
            pcie = metric(jobs['seed_pcie'])
            status[f'seed{seed}'] = {'near_tpot': metric(near),
                                     'fast_tpot': fast, 'pcie_tpot': pcie,
                                     'pcie_gain_vs_fast_pct': 100 * (fast - pcie) / fast,
                                     'teacher_sha256': sha(teacher)}
            write(status)
        best = max(SEEDS['seeds'], key=lambda s: status[f'seed{s}']['pcie_gain_vs_fast_pct'])
        status['best_observed_seed'] = best
        write(status)
        teacher = RAW / f'seed{best}' / 'TEACHER_TOKENS.json'
        confirmations = {}
        for label, policy in (('confirm_fast', 'NEAR_FAST'), ('confirm_pcie', 'NEAR_PCIE')):
            first = run(best, label, policy, 1, repeats=2, teacher=teacher)
            rows = [metric(first, index=i) for i in (1, 2)]
            e2e = [metric(first, 'E2E', i) for i in (1, 2)]
            gaps = {'TPOT': 100 * abs(rows[0] - rows[1]) / statistics.mean(rows),
                    'E2E': 100 * abs(e2e[0] - e2e[1]) / statistics.mean(e2e)}
            if max(gaps.values()) > 2:
                extra = run(best, label, policy, 2, teacher=teacher)
                rows.append(metric(extra))
                e2e.append(metric(extra, 'E2E'))
            confirmations[label] = {'TPOT_samples': rows, 'E2E_samples': e2e,
                                    'first_pair_gap_pct': gaps,
                                    'reported': statistics.median(rows) if len(rows) == 3
                                    else statistics.mean(rows)}
            status['confirmation'] = confirmations
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
