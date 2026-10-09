"""Summarize complete, unfiltered R2/R8 Qwen main-table receipts."""

import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOTS = {
    2: Path('/home/hwlee/mgo-results/qwen_r2_sharegpt_b16_l512_20261010'),
    8: Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009'),
}
SYSTEMS = ('ours', 'deepspeed', 'infinity', 'llama')
METRICS = ('TTFT', 'TPOT', 'E2E', 'throughput')


def job(root, system):
    candidates = sorted((root / 'jobs').glob(f'{system}_full_v*'),
                        key=lambda p: int(p.name.rsplit('_v', 1)[1]), reverse=True)
    for path in candidates:
        status_file = path / 'status.json'
        if status_file.exists() and json.loads(status_file.read_text())['status'] == 'PASS':
            return path
    raise FileNotFoundError(f'no PASS full job for {system} under {root}')


def read_row(world, system):
    root = ROOTS[world]
    path = job(root, system)
    status = json.loads((path / 'status.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    assert result['status'] == 'PASS' and not status['smoke']
    assert result.get('headline_eligible', True)
    assert status['physical_gpus'] == ([0, 1] if world == 2 else list(range(8)))
    assert status['repeats'] == 2 and status['quiet_2367']
    assert len(list(path.glob('repeat[1-9].json'))) == 2
    samples = [json.loads((path / f'repeat{i}.json').read_text()) for i in (1, 2)]
    assert all(row['status'] == 'PASS' and row['output_tokens'] == 64
               and row['global_requests'] == world * 16 for row in samples)
    if system == 'ours':
        command = status['command']
        assert all(flag in command for flag in ('--expert-executor', 'native',
                    '--prefetch-off', '--policy', 'LA_CA_NEAR',
                    '--prefill-layout-fast', '--decode-layout-fast'))
        assert result['expert_executor'] == 'native' and result['policy'] == 'LA_CA_NEAR'
    if system == 'llama':
        audit = json.loads((path / 'placement_audit.json').read_text())
        assert audit['status'] == 'PASS'
        assert sorted(audit['gpu_expert_layers_by_device'].values()) == \
            ([7] * 2 if world == 2 else [1] * 8)
    metrics = {}
    for metric in METRICS:
        values = [(world * 16 * 64 / float(row['E2E'])) if metric == 'throughput'
                  else float(row[metric]) for row in samples]
        mean = statistics.mean(values)
        metrics[metric] = dict(samples=values, mean=mean, minimum=min(values),
                               maximum=max(values), relative_range_pct=100 *
                               (max(values) - min(values)) / mean)
    unstable = max(metrics[key]['relative_range_pct'] for key in ('TPOT', 'E2E')) > 5
    third_advised = not unstable and max(metrics[key]['relative_range_pct']
                                         for key in ('TPOT', 'E2E')) > 2
    return dict(path=str(path), source_commit=status['source_commit'],
                workload_sha256=status['workload_sha256'],
                physical_gpus=status['physical_gpus'], system=result['system'],
                metrics=metrics, unstable=unstable,
                third_advised=third_advised)


def fmt(row, key):
    metric = row['metrics'][key]
    return f"{metric['mean']:.3f} [{metric['minimum']:.3f}, {metric['maximum']:.3f}]"


def main():
    data = {str(world): {system: read_row(world, system) for system in SYSTEMS}
            for world in (2, 8)}
    (HERE / 'RESULTS.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Qwen ShareGPT R2/R8 main table', '',
             'Input 512, 64 greedy output tokens, B16 per GPU, C30 expert '
             'budget. TPOT is seconds per decoded token and includes attention; '
             'TPS is global output tokens divided by E2E seconds. Values are '
             'the mean and full range of the two unfiltered target repeats '
             'after warmup. R2 and R8 have different global batches (32 and '
             '128), so throughput is not a same-batch scaling comparison.', '',
             '| Ranks | System | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Repeat quality |',
             '|---:|---|---:|---:|---:|---:|---|']
    for world in (2, 8):
        for system in SYSTEMS:
            row = data[str(world)][system]
            quality = ('unstable (>5% TPOT/E2E range)' if row['unstable'] else
                       'needs third (>2% range)' if row['third_advised'] else '≤2% TPOT/E2E range')
            lines.append(f"| {world} | {row['system']} | {fmt(row, 'TTFT')} | "
                         f"{fmt(row, 'TPOT')} | {fmt(row, 'E2E')} | "
                         f"{fmt(row, 'throughput')} | {quality} |")
    lines += ['', 'Both target repeats are shown without outlier selection. '
              'The quality flag uses the relative full range of TPOT and E2E; '
              'TTFT variability is visible separately in its range. Raw paths, '
              'source commits, and workload hashes are in [RESULTS.json](RESULTS.json).', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
