"""Audit the bounded grouped R8 sample-seed screen and its confirmation."""

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/pcie_quota_grouped_r8_b16_20261010/seeds')
POLICIES = {'seed_near': 'LA_CA_NEAR', 'seed_fast': 'NEAR_FAST',
            'seed_pcie': 'NEAR_PCIE', 'confirm_fast': 'NEAR_FAST',
            'confirm_pcie': 'NEAR_PCIE'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_job(seed, label, attempt, manifest_hash, teacher_hash=None):
    path = RAW / f'seed{seed}/jobs/{label}_full_v{attempt}'
    status = json.loads((path / 'status.json').read_text())
    assert status['status'] == 'PASS' and status['ours_mode'] == 'N'
    assert status['physical_gpus'] == list(range(8))
    assert status['workload_sha256'] == manifest_hash
    command = status['command']
    assert command[command.index('--policy') + 1] == POLICIES[label]
    assert command[command.index('--grouped-decode-mode') + 1] == 'hit_then_miss'
    assert '--compiled-dense' in command and '--prefetch-off' in command
    if teacher_hash is not None:
        assert sha(Path(command[command.index('--teacher-tokens') + 1])) == teacher_hash
    repeats = status['repeats']
    metrics = []
    teacher = (json.loads((RAW / f'seed{seed}/TEACHER_TOKENS.json').read_text())
               if teacher_hash is not None else None)
    for repeat in range(1, repeats + 1):
        row = json.loads((path / f'repeat{repeat}.json').read_text())
        ranks = [json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                 for rank in range(8)]
        assert all(rank['phase'] == 'target' and rank['policy'] == POLICIES[label]
                   and rank['expert_cache_start'] == 'empty'
                   and rank['grouped_decode_mode'] == 'hit_then_miss'
                   and rank['compiled_dense'] and rank['prefetch_off']
                   and rank['validation']['status'] == 'PASS'
                   and rank['validation']['controller']['quota_violations'] == 0
                   and len(rank['tokens']) == 16
                   and all(len(sequence) == 64 for sequence in rank['tokens'])
                   for rank in ranks)
        if teacher_hash is not None:
            assert all(rank['forced_continuation'] for rank in ranks)
            assert all(rank['request_ids'] == teacher['request_ids'][str(index)]
                       for index, rank in enumerate(ranks))
        metric = {key: float(row[key]) for key in ('TTFT', 'TPOT', 'E2E')}
        if teacher is not None:
            metric['token_differences_vs_teacher'] = sum(
                actual != expected
                for index, rank in enumerate(ranks)
                for actual_sequence, expected_sequence in zip(
                    rank['tokens'], teacher['rank_tokens'][str(index)])
                for actual, expected in zip(actual_sequence, expected_sequence))
        metrics.append(metric)
    return {'path': str(path), 'source_commit': status['source_commit'],
            'metrics': metrics}


def main():
    seeds = json.loads((HERE / 'SEEDS.json').read_text())
    status = json.loads((HERE / 'SEED_STATUS.json').read_text())
    assert status['status'] == 'PASS' and seeds['seeds'] == status['seeds']
    rows = []
    commits = set()
    for candidate in seeds['candidates']:
        seed = candidate['seed']
        manifest = Path(candidate['manifest'])
        assert sha(manifest) == candidate['manifest_sha256']
        teacher = RAW / f'seed{seed}/TEACHER_TOKENS.json'
        teacher_hash = sha(teacher)
        assert teacher_hash == status[f'seed{seed}']['teacher_sha256']
        jobs = {label: audit_job(seed, label, 1, sha(manifest),
                                 None if label == 'seed_near' else teacher_hash)
                for label in ('seed_near', 'seed_fast', 'seed_pcie')}
        for job in jobs.values():
            commits.add(job['source_commit'])
        near = jobs['seed_near']['metrics'][0]['TPOT']
        fast = jobs['seed_fast']['metrics'][0]['TPOT']
        pcie = jobs['seed_pcie']['metrics'][0]['TPOT']
        gain = 100 * (fast - pcie) / fast
        assert abs(gain - status[f'seed{seed}']['pcie_gain_vs_fast_pct']) < 1e-8
        rows.append({'seed': seed, 'near_tpot': near, 'fast_tpot': fast,
                     'pcie_tpot': pcie, 'pcie_gain_vs_fast_pct': gain,
                     'jobs': jobs, 'teacher_sha256': teacher_hash})
    assert len(commits) == 1
    best = max(rows, key=lambda row: row['pcie_gain_vs_fast_pct'])
    assert best['seed'] == status['best_observed_seed']
    seed = best['seed']
    manifest_hash = sha(RAW / f'seed{seed}/WORKLOADS.json')
    confirmation = {}
    for label in ('confirm_fast', 'confirm_pcie'):
        jobs = [audit_job(seed, label, 1, manifest_hash, best['teacher_sha256'])]
        assert len(jobs[0]['metrics']) == 2
        if len(status['confirmation'][label]['TPOT_samples']) == 3:
            jobs.append(audit_job(seed, label, 2, manifest_hash,
                                  best['teacher_sha256']))
        commits.update(job['source_commit'] for job in jobs)
        samples = [metric for job in jobs for metric in job['metrics']]
        values = [metric['TPOT'] for metric in samples]
        assert values == status['confirmation'][label]['TPOT_samples']
        confirmation[label] = {'jobs': jobs, 'samples': samples,
                               'reported_tpot': statistics.median(values)
                               if len(values) == 3 else statistics.mean(values),
                               'minimum_tpot': min(values),
                               'maximum_tpot': max(values)}
    fast = confirmation['confirm_fast']['reported_tpot']
    pcie = confirmation['confirm_pcie']['reported_tpot']
    confirmed_gain = 100 * (fast - pcie) / fast
    assert len(commits) == 1
    paired_token_differences = 0
    fast_path = Path(confirmation['confirm_fast']['jobs'][0]['path'])
    pcie_path = Path(confirmation['confirm_pcie']['jobs'][0]['path'])
    for rank in range(8):
        fast_tokens = json.loads((fast_path / f'repeat1_rank{rank}.json').read_text())['tokens']
        pcie_tokens = json.loads((pcie_path / f'repeat1_rank{rank}.json').read_text())['tokens']
        paired_token_differences += sum(
            left != right for fast_sequence, pcie_sequence in
            zip(fast_tokens, pcie_tokens) for left, right in
            zip(fast_sequence, pcie_sequence))
    result = {'status': 'PASS', 'source_commit': commits.pop(),
              'screen': rows, 'best_observed_seed': seed,
              'confirmation': confirmation,
              'confirmed_pcie_gain_vs_fast_pct': confirmed_gain,
              'first_confirmation_predicted_token_differences': paired_token_differences}
    (HERE / 'SEED_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Bounded grouped R8 ShareGPT sample-seed screen', '',
             'Qwen3-30B, R8/C30/local-B16/input512/output64, strict '
             'hit-then-miss `new_OURS`, compiled dense, prefetch OFF. '
             'The four seeds were frozen before timing; each screen cell has '
             'one unfiltered target, so the screen is exploratory. The same '
             'continuation token IDs are supplied to fast-rank and PCIe '
             'policies, while their BF16 routing and predicted tokens can '
             'differ from the Near teacher; counts are in the JSON.', '',
             '| Seed | Near TPOT | Fast-rank TPOT | PCIe-lookup TPOT | '
             'PCIe gain vs fast |',
             '|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['seed']} | {row['near_tpot']:.4f} | "
                     f"{row['fast_tpot']:.4f} | {row['pcie_tpot']:.4f} | "
                     f"{row['pcie_gain_vs_fast_pct']:+.2f}% |")
    lines += ['', f'Seed {seed} had the largest **observed single-shot** PCIe '
              'gain. Fresh confirmations on that seed:', '',
              '| Policy | Repeats | TPOT center (s/token) | Full TPOT range |',
              '|---|---:|---:|---:|']
    for label, name in (('confirm_fast', 'Fast-rank quota'),
                        ('confirm_pcie', 'PCIe lookup quota')):
        row = confirmation[label]
        lines.append(f"| {name} | {len(row['samples'])} | "
                     f"{row['reported_tpot']:.4f} | "
                     f"[{row['minimum_tpot']:.4f}, {row['maximum_tpot']:.4f}] |")
    lines += ['', f'Confirmed PCIe gain versus fast-rank quota: '
              f'**{confirmed_gain:+.2f}%** on the selected seed. '
              'The full TPOT ranges overlap, and the original fixed workload '
              'showed a 0.60% regression, so a reliable lookup advantage is '
              'not established. Keep the current main_OURS Near default. '
              f'The first Fast/PCIe confirmation predictions differed at '
              f'{paired_token_differences} of 8,192 output-token positions '
              'despite the same supplied continuation IDs. '
              'The selected seed is not a claim of a population optimum. '
              'All raw attempts and metrics are retained in '
              '[SEED_RESULTS.json](SEED_RESULTS.json).', '']
    (HERE / 'SEED_RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
