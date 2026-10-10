"""Summarize complete, unfiltered R2/R8 Qwen main-table receipts."""

import json
import hashlib
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
ROOTS = {
    2: Path('/home/hwlee/mgo-results/qwen_r2_sharegpt_b16_l512_20261010'),
    8: Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009'),
}
SYSTEMS = ('ours', 'deepspeed', 'infinity', 'llama')
METRICS = ('TTFT', 'TPOT', 'E2E', 'throughput')


def job(root, system, repeats, after=0, before=None):
    candidates = sorted((root / 'jobs').glob(f'{system}_full_v*'),
                        key=lambda p: int(p.name.rsplit('_v', 1)[1]), reverse=True)
    for path in candidates:
        status_file = path / 'status.json'
        attempt = int(path.name.rsplit('_v', 1)[1])
        if not status_file.exists() or attempt <= after or (before and attempt >= before):
            continue
        status = json.loads(status_file.read_text())
        receipt = path / 'result.json'
        if (status['status'] == 'PASS' and not status['smoke'] and
                status['repeats'] == repeats and receipt.exists() and
                json.loads(receipt.read_text()).get('headline_eligible', True)):
            return path
    raise FileNotFoundError(f'no eligible {repeats}-repeat job for {system} under {root}')


def read_row(world, system):
    root = ROOTS[world]
    manifest_path = root / 'WORKLOADS.json'
    manifest = json.loads(manifest_path.read_text())
    cell = manifest['cells'][0]
    assert manifest['status'] == 'FROZEN'
    assert cell['global_requests'] == world * 16
    assert cell['input_tokens'] == 512 and cell['output_tokens'] == 64
    assert cell['cache_percent'] == 30
    assert sum(cell['expert_slots_per_rank']) == 1843
    path = job(root, system, 2)
    status = json.loads((path / 'status.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    assert result['status'] == 'PASS' and not status['smoke']
    assert result.get('headline_eligible', True)
    assert status['physical_gpus'] == ([0, 1] if world == 2 else list(range(8)))
    assert status['repeats'] == 2 and status['quiet_2367']
    assert status['workload_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
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
    if system == 'infinity':
        config = json.loads((path / 'config.json').read_text())
        assert config['speculative_admission'] == 'disabled'
        assert config['token_progress_receipt']
        assert result['speculative_admission'] == 'disabled'
    if system == 'llama':
        audit = json.loads((path / 'placement_audit.json').read_text())
        assert audit['status'] == 'PASS'
        assert sorted(audit['gpu_expert_layers_by_device'].values()) == \
            ([7] * 2 if world == 2 else [1] * 8)
    pair_ranges = {}
    for key in ('TPOT', 'E2E'):
        values = [float(row[key]) for row in samples]
        pair_ranges[key] = 100 * (max(values) - min(values)) / statistics.mean(values)
    unstable = max(pair_ranges.values()) > 5
    third_advised = max(pair_ranges.values()) > 2
    paths = [path]
    if third_advised:
        extra = job(root, system, 1, after=int(path.name.rsplit('_v', 1)[1]),
                    before=3 if world == 2 and system == 'ours' else None)
        extra_status = json.loads((extra / 'status.json').read_text())
        assert extra_status['workload_sha256'] == status['workload_sha256']
        assert extra_status['physical_gpus'] == status['physical_gpus']
        assert extra_status['quiet_2367']
        extra_row = json.loads((extra / 'repeat1.json').read_text())
        assert extra_row['status'] == 'PASS' and extra_row['output_tokens'] == 64
        assert extra_row['global_requests'] == world * 16
        samples.append(extra_row)
        paths.append(extra)
    confirmation_path = None
    confirmation_row = None
    if world == 2 and system == 'ours':
        confirmation_path = job(root, system, 1, after=2, before=4)
        confirmation_status = json.loads((confirmation_path / 'status.json').read_text())
        assert confirmation_status['workload_sha256'] == status['workload_sha256']
        assert confirmation_status['physical_gpus'] == status['physical_gpus']
        assert confirmation_status['quiet_2367']
        confirmation_row = json.loads((confirmation_path / 'repeat1.json').read_text())
        assert confirmation_row['status'] == 'PASS'
        assert confirmation_row['output_tokens'] == 64
        assert confirmation_row['global_requests'] == world * 16
    for source in paths + ([confirmation_path] if confirmation_path else []):
        source_status = json.loads((source / 'status.json').read_text())
        for repeat in range(1, source_status['repeats'] + 1):
            if system == 'ours':
                for rank in range(world):
                    receipt = json.loads((source / f'repeat{repeat}_rank{rank}.json').read_text())
                    assert receipt['expert_executor'] == 'native'
                    assert receipt['policy'] == 'LA_CA_NEAR'
                    assert receipt['prefetch_off'] and receipt['native_prefill']
                    assert receipt['prefill_layout_fast'] and receipt['decode_layout_fast']
                    assert receipt['grouped_decode_mode'] == 'off'
                    assert receipt['expert_cache_start'] == 'empty'
                    assert receipt['no_compile'] and receipt['validation']['status'] == 'PASS'
                    assert receipt['validation']['controller']['issued'] == 0
                    assert receipt['validation']['physical_slots'] == 1843
                    assert receipt['validation']['main_slots'] == 1843 - 2 * world
            elif system == 'deepspeed':
                for rank in range(world):
                    receipt = json.loads((source / f'repeat{repeat}_rank{rank}.json').read_text())
                    assert receipt['rank'] == rank
                    assert receipt['all_parameter_peak_bytes'] <= receipt['parameter_budget_bytes']
                    assert receipt['kv_gpu_resident'] and receipt['finite_logits']
            elif system == 'infinity':
                receipt = json.loads((source / f'repeat{repeat}.json').read_text())
                assert receipt['eam_candidates'] == 0 and receipt['kv_released']
                assert receipt['cache_after']['peak_accounted_bytes'] <= sum(receipt['expert_budget_per_gpu'])
            else:
                config = json.loads((source / 'config.json').read_text())
                assert config['expert_resident_bytes'] <= config['expert_budget_bytes']
                assert config['synchronous_batch'] and config['cuda_graphs_runtime'] == 'off'
    metrics = {}
    for metric in METRICS:
        values = [(world * 16 * 64 / float(row['E2E'])) if metric == 'throughput'
                  else float(row[metric]) for row in samples]
        mean = statistics.mean(values)
        metrics[metric] = dict(samples=values, mean=mean,
                               reported=statistics.median(values) if len(values) == 3 else mean,
                               minimum=min(values),
                               maximum=max(values), relative_range_pct=100 *
                               (max(values) - min(values)) / mean)
    confirmation = None
    if confirmation_row is not None:
        confirmation = dict(path=str(confirmation_path),
                            source_commit=confirmation_status['source_commit'],
                            metrics={key: (world * 16 * 64 / float(confirmation_row['E2E'])
                                           if key == 'throughput' else float(confirmation_row[key]))
                                     for key in METRICS},
                            relative_to_primary_median_pct={
                                key: 100 * (float(confirmation_row[key]) - metrics[key]['reported']) /
                                metrics[key]['reported'] for key in ('TTFT', 'TPOT', 'E2E')})
    return dict(paths=list(map(str, paths)),
                source_commits=[json.loads((p / 'status.json').read_text())['source_commit'] for p in paths],
                workload_sha256=status['workload_sha256'],
                physical_gpus=status['physical_gpus'], system=result['system'],
                metrics=metrics, unstable=unstable,
                third_repeat_used=third_advised, pair_relative_ranges_pct=pair_ranges,
                confirmation=confirmation)


def fmt(row, key):
    metric = row['metrics'][key]
    places = 4 if key == 'TPOT' else 3
    return (f"{metric['reported']:.{places}f} "
            f"[{metric['minimum']:.{places}f}, {metric['maximum']:.{places}f}]")


def main():
    data = {str(world): {system: read_row(world, system) for system in SYSTEMS}
            for world in (2, 8)}
    (HERE / 'RESULTS.json').write_text(json.dumps(data, indent=2) + '\n')
    lines = ['# Qwen ShareGPT R2/R8 main table', '',
             'Input 512, 64 greedy output tokens, B16 per GPU, C30 expert '
             'budget. TPOT is seconds per decoded token and includes attention; '
             'TPS is global output tokens divided by E2E seconds. Each cell '
             'has two unfiltered target repeats after warmup; if the pair '
             'differs by >2% in TPOT or E2E, exactly one third target '
             'is added. The reported center is the mean of two or median of '
             'three, followed by the full range. An initial gap above 5% '
             'remains flagged even after the third repeat. R2 and R8 have different '
             'global batches (32 and '
             '128), so throughput is not a same-batch scaling comparison. '
             'MoE-Infinity uses EAM eviction priorities with speculative '
             'transfer admission disabled on both rank counts after the '
             'R8 long-target native wait stall; the expert budget is unchanged.', '',
             '| Ranks | System | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Repeat quality |',
             '|---:|---|---:|---:|---:|---:|---|']
    for world in (2, 8):
        for system in SYSTEMS:
            row = data[str(world)][system]
            quality = ('initial pair >5%; third repeat, median shown' if row['unstable'] else
                       'third repeat; median shown' if row['third_repeat_used'] else '≤2% TPOT/E2E pair range')
            lines.append(f"| {world} | {row['system']} | {fmt(row, 'TTFT')} | "
                         f"{fmt(row, 'TPOT')} | {fmt(row, 'E2E')} | "
                         f"{fmt(row, 'throughput')} | {quality} |")
    r2_confirmation = data['2']['ours']['confirmation']
    lines += ['', 'The separately requested fourth R2 main_OURS run is a '
              'confirmation, not part of the original three-run primary '
              'median: TTFT '
              f"{r2_confirmation['metrics']['TTFT']:.3f} s, TPOT "
              f"{r2_confirmation['metrics']['TPOT']:.4f} s/token, E2E "
              f"{r2_confirmation['metrics']['E2E']:.3f} s, TPS "
              f"{r2_confirmation['metrics']['throughput']:.3f}. "
              'Its raw receipt and exact comparison with the primary '
              'median are in [RESULTS.json](RESULTS.json).', '',
              'R8 main_OURS had one faster middle run. Its first and third '
              'runs differed by 0.27% in TPOT and 0.43% in E2E; the '
              'three-run median is close to those two, while the full range '
              'remains shown above.', '',
              'All target repeats are preserved without outlier selection '
              'in [RESULTS.json](RESULTS.json). '
              'The quality flag uses the relative difference of the first two '
              'TPOT and E2E values; '
              'TTFT variability is visible separately in its range. Raw paths, '
              'source commits, and workload hashes are in the same JSON file.', '',
              'main_OURS has the lowest E2E in both rank counts. At R2, '
              'llama.cpp has the lowest TPOT but its approximately 74-second '
              'TTFT makes its E2E longer than main_OURS. At R8, main_OURS '
              'has the lowest TPOT as well; its initial E2E pair exceeded '
              '5%, so the three-repeat median must retain its full range.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
