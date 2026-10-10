"""Audit and summarize the grouped matched-token quota targets."""

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
TEACHER = HERE.parent / 'pcie_quota_r8_b16_20261010/FROZEN_TOKENS.json'
TABLE = HERE.parent / 'pcie_quota_r8_b16_20261010/LOOKUP.json'
POLICIES = (('group_near', 'LA_CA_NEAR', 'Original Near'),
            ('group_fast', 'NEAR_FAST', 'Fast-rank quota'),
            ('group_pcie', 'NEAR_PCIE', 'PCIe lookup quota'))
METRICS = ('TTFT', 'TPOT', 'E2E', 'throughput')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(label, policy, name, teacher):
    first = RAW / 'jobs' / f'{label}_full_v1'
    status = json.loads((first / 'status.json').read_text())
    assert status['status'] == 'PASS' and status['repeats'] == 2
    assert status['ours_mode'] == 'N' and status['physical_gpus'] == list(range(8))
    assert status['quiet_2367'] and status['workload_sha256'] == teacher['workload_sha256']
    command = status['command']
    assert command[command.index('--policy') + 1] == policy
    assert command[command.index('--grouped-decode-mode') + 1] == 'hit_then_miss'
    assert '--compiled-dense' in command and '--prefetch-off' in command
    assert sha(Path(command[command.index('--teacher-tokens') + 1])) == sha(TEACHER)
    if policy != 'LA_CA_NEAR':
        assert sha(Path(command[command.index('--quota-table') + 1])) == sha(TABLE)
    paths = [first]
    samples = [json.loads((first / f'repeat{i}.json').read_text()) for i in (1, 2)]
    gaps = {metric: 100 * abs(samples[0][metric] - samples[1][metric]) /
            statistics.mean(row[metric] for row in samples)
            for metric in ('TPOT', 'E2E')}
    if max(gaps.values()) > 2:
        extra = RAW / 'jobs' / f'{label}_full_v2'
        extra_status = json.loads((extra / 'status.json').read_text())
        assert extra_status['status'] == 'PASS' and extra_status['repeats'] == 1
        assert extra_status['source_commit'] == status['source_commit']
        paths.append(extra)
        samples.append(json.loads((extra / 'repeat1.json').read_text()))
    audits = []
    for path in paths:
        repeats = json.loads((path / 'status.json').read_text())['repeats']
        for repeat in range(1, repeats + 1):
            ranks = [json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                     for rank in range(8)]
            assert all(row['phase'] == 'target' and row['expert_cache_start'] == 'empty'
                       and row['policy'] == policy and row['forced_continuation']
                       and row['grouped_decode_mode'] == 'hit_then_miss'
                       and row['compiled_dense'] and row['prefetch_off']
                       and row['validation']['status'] == 'PASS'
                       and row['validation']['controller']['quota_violations'] == 0
                       for row in ranks)
            differences = [sum(a != b for actual, expected in
                               zip(row['tokens'], teacher['rank_tokens'][str(rank)])
                               for a, b in zip(actual, expected))
                           for rank, row in enumerate(ranks)]
            audits.append({'job': str(path), 'repeat': repeat,
                           'actual_token_differences_vs_teacher': differences,
                           'mandatory_misses': ranks[0]['validation']['controller']['mandatory'],
                           'h2d_copies': [row['validation']['scheduler']['copies'] for row in ranks],
                           'h2d_bytes': [row['validation']['scheduler']['bytes'] for row in ranks]})
    metrics = {}
    for metric in METRICS:
        values = [float(sample[metric]) for sample in samples]
        metrics[metric] = {'samples': values,
                           'reported': statistics.median(values) if len(values) == 3
                           else statistics.mean(values),
                           'minimum': min(values), 'maximum': max(values)}
    return {'name': name, 'policy': policy, 'source_commit': status['source_commit'],
            'first_pair_gap_pct': gaps, 'metrics': metrics, 'audits': audits}


def fmt(row, metric):
    value = row['metrics'][metric]
    digits = 4 if metric == 'TPOT' else 3
    return (f"{value['reported']:.{digits}f} "
            f"[{value['minimum']:.{digits}f}, {value['maximum']:.{digits}f}]")


def main():
    assert json.loads((HERE / 'STATUS.json').read_text())['status'] == 'PASS'
    teacher = json.loads(TEACHER.read_text())
    rows = [load(label, policy, name, teacher) for label, policy, name in POLICIES]
    assert len({row['source_commit'] for row in rows}) == 1
    data = {'status': 'PASS', 'workload': teacher['cell'],
            'teacher_sha256': sha(TEACHER), 'table_sha256': sha(TABLE),
            'policies': rows}
    (HERE / 'GROUPED_RESULTS.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# R8 B16 grouped new_OURS PCIe quota comparison', '',
             'Qwen3-30B / ShareGPT / C30 / input512 / output64 / prefetch OFF. '
             'All policies use native prefill, compiled dense routing and strict '
             'hit-then-miss grouped decode. The same next-token IDs are supplied '
             'to each policy; BF16 internal routing can still differ.', '',
             '| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | '
             'First-run actual token differences |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        differences = sum(row['audits'][0]['actual_token_differences_vs_teacher'])
        lines.append(f"| {row['name']} | {len(row['metrics']['TPOT']['samples'])} | "
                     f"{fmt(row, 'TTFT')} | {fmt(row, 'TPOT')} | "
                     f"{fmt(row, 'E2E')} | {fmt(row, 'throughput')} | {differences} |")
    lines += ['', 'Each reported value is the mean of two or median of three '
              'unfiltered repeats; brackets are the full range. The third '
              'repeat rule uses the first-pair TPOT or E2E gap >2%. '
              'See [GROUPED_RESULTS.json](GROUPED_RESULTS.json) for all samples, '
              'rank fetch counts and raw receipt paths.', '']
    fast = rows[1]['metrics']['TPOT']['reported']
    pcie = rows[2]['metrics']['TPOT']['reported']
    delta = 100 * (fast - pcie) / fast
    lines += [f'The PCIe lookup is {abs(delta):.2f}% '
              f'{"faster" if delta > 0 else "slower"} than the fast-rank quota '
              'on the reported TPOT. Their full ranges overlap, so this does '
              'not establish a lookup gain. The fixed four-seed screen is a '
              'separate exploratory follow-up; selection on one timing sample '
              'per policy cannot establish a population-best seed.', '']
    (HERE / 'GROUPED_RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
