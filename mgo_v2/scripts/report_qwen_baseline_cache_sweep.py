"""Validate and summarize Qwen baseline cache-sweep receipts."""

import hashlib
import json
import statistics
from pathlib import Path


P = Path(__file__).resolve().parents[1]
OUT = P / 'experiments/qwen_cache_ablation_20261009'
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
SOURCE = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
OLD_C30 = {
    'infinity': 'mt2_qwen_sharegpt_b16_l512_infinity_r3_v3',
    'deepspeed': 'mt2_qwen_sharegpt_b16_l512_deepspeed_r3_v1',
    'llama': 'mt2_qwen_sharegpt_b16_l512_llama_r3_v1',
}
SYSTEMS = ('infinity', 'deepspeed', 'llama')
WORKERS = {
    'infinity': 'headline_infinity_worker.py',
    'deepspeed': 'headline_deepspeed_worker.py',
    'llama': 'headline_llama_sync_worker.py',
}


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return dict(samples=values, mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values),
                maximum=max(values), stdev=statistics.stdev(values))


def find_job(cap, system):
    if cap == 30:
        return JOBS / OLD_C30[system], True
    paths = sorted(JOBS.glob(f'qca_baseline_c{cap}_{system}_r3_v*'),
                   key=lambda p: int(p.name.rsplit('_v', 1)[1]))
    passed = [p for p in paths if (p / 'status.json').exists() and
              read(p / 'status.json').get('status') == 'PASS']
    return (passed[-1] if passed else None), False


def inspect(spec, system, job, reused, original_sha, current_sha):
    status = read(job / 'status.json')
    assert status['status'] == 'PASS'
    assert status['workload_manifest_sha256'] == (original_sha if reused else current_sha)
    assert any(str(arg).endswith(WORKERS[system]) for arg in status['command'])
    assert '--repeats' in status['command'] and status['command'][status['command'].index('--repeats') + 1] == '3'
    assert reused or status.get('min_gpu_free_mib') == 2048
    assert not status.get('remaining_gpu_occupants')
    assert reused or {r['gpu'] for r in status['restored']} == {0, 1, 4, 5}
    source_spec = next(c for c in read(SOURCE)['cells'] if c['cell'] == spec['cell'])
    for phase in ('warmup', 'target'):
        assert spec[phase]['sha256'] == source_spec[phase]['sha256']
        assert hashlib.sha256(Path(spec[phase]['path']).read_bytes()).hexdigest() == spec[phase]['sha256']
    expected_ids = [row['request_id'] for row in read(Path(spec['target']['path']))['requests']]
    repeats = [read(job / f'repeat{i}.json') for i in (1, 2, 3)]
    assert all(r['status'] == 'PASS' and r['output_tokens'] == 64 and r['global_requests'] == 64 for r in repeats)
    memory = {}
    for i, record in enumerate(repeats, 1):
        if system == 'infinity':
            assert record['request_ids'] == expected_ids
            assert all(len(tokens) == 64 for tokens in record['tokens'])
            budgets = record['expert_budget_per_gpu']
            peaks = [record['cache_after'][f'gpu_{r}_peak_charged_bytes'] for r in range(4)]
            assert all(0 <= peak <= budget for peak, budget in zip(peaks, budgets))
            memory.setdefault('peak_expert_bytes_per_gpu', []).append(peaks)
            memory['expert_budget_bytes_per_gpu'] = budgets
        elif system == 'deepspeed':
            rr = [read(job / f'repeat{i}_rank{r}.json') for r in range(4)]
            assert [rid for row in rr for rid in row['request_ids']] == expected_ids
            assert all(len(tokens) == 64 for row in rr for tokens in row['tokens'])
            assert all(0 <= row['all_parameter_peak_bytes'] <= row['parameter_budget_bytes'] for row in rr)
            memory.setdefault('peak_all_parameter_bytes_per_rank', []).append([row['all_parameter_peak_bytes'] for row in rr])
            memory['parameter_budget_bytes_per_rank'] = [row['parameter_budget_bytes'] for row in rr]
        else:
            assert record['request_ids'] == expected_ids
            assert all(len(tokens) == 64 for tokens in record['tokens'])
            per = {20: 2, 30: 3, 40: 4, 50: 6}[spec['cache_percent']]
            assert record['synchronous_batch'] and record['expert_placement'] == f'balanced{per}'
            assert record['gpu_expert_layers_by_device'] == {f'CUDA{r}': per for r in range(4)}
            assert record['expert_resident_bytes'] <= spec['expert_budget_bytes']
            placement = read(job / 'placement_audit.json')
            assert placement['gpu_expert_layers_by_device'] == record['gpu_expert_layers_by_device']
            assert placement['gpu_expert_bytes'] == record['expert_resident_bytes']
            cfg = read(job / 'config.json')
            assert cfg['cuda_graphs_runtime'] == 'off' and not cfg['llama_graph_reuse']
            assert cfg['cpu_threads'] == 32 and not cfg['op_offload']
            memory['resident_expert_bytes_per_gpu'] = [record['expert_resident_bytes'] // 4] * 4
            memory['expert_budget_bytes_global'] = spec['expert_budget_bytes']
    return dict(status='PASS', system=system, cache_percent=spec['cache_percent'],
                label=job.name, reused_c30=reused, source_commit=status['source_commit'],
                workload_manifest_sha256=status['workload_manifest_sha256'],
                target_sha256=spec['target']['sha256'],
                metrics={name: stats([r[name] for r in repeats]) for name in ('TTFT', 'TPOT', 'E2E')},
                memory=memory)


def main():
    manifest = read(MANIFEST)
    current_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    original_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == [0, 1, 4, 5]
    rows = []
    for spec in sorted(manifest['cells'], key=lambda c: c['cache_percent']):
        for system in SYSTEMS:
            job, reused = find_job(spec['cache_percent'], system)
            rows.append(inspect(spec, system, job, reused, original_sha, current_sha)
                        if job else dict(status='PENDING', system=system,
                                         cache_percent=spec['cache_percent']))
    result = dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'PARTIAL',
                  completed=sum(r['status'] == 'PASS' for r in rows), total=len(rows),
                  current_manifest_sha256=current_sha, original_manifest_sha256=original_sha,
                  physical_gpus=[0, 1, 4, 5], rows=rows)
    (OUT / 'BASELINE_SWEEP_RESULTS.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Qwen cache-capacity baseline sweep', '',
             'ShareGPT, R4 GPUs 0/1/4/5, local B16/input512/output64. Three unfiltered clean target repeats per baseline cell. C30 is validated reuse of the earlier same-request, same-budget run; C20/C40/C50 are new guarded jobs.', '',
             '| Cache | System | TTFT median [range], s | TPOT median [range], s/token | E2E median [range], s | Receipt |',
             '|---:|---|---:|---:|---:|---|']
    for row in rows:
        if row['status'] != 'PASS':
            lines.append(f"| C{row['cache_percent']} | {row['system']} | pending | pending | pending | — |")
            continue
        def fmt(name):
            x = row['metrics'][name]
            return f"{x['median']:.3f} [{x['minimum']:.3f}, {x['maximum']:.3f}]"
        receipt = row['label'] + (' (C30 reuse)' if row['reused_c30'] else '')
        lines.append(f"| C{row['cache_percent']} | {row['system']} | {fmt('TTFT')} | {fmt('TPOT')} | {fmt('E2E')} | `{receipt}` |")
    lines += ['', 'Expert residency/placement or live-parameter budget checks are recorded per cell in `BASELINE_SWEEP_RESULTS.json`. llama.cpp statically holds balanced full expert layers, so its quantized GPU residency is not a dynamic cache hit-rate measurement. DeepSpeed limits all live parameters, including non-experts, under its cap. Raw requests, tokens, logs and GPU resource samples remain outside Git.', '']
    (OUT / 'BASELINE_SWEEP_RESULTS.md').write_text('\n'.join(lines))
    print(f"{result['status']} {result['completed']}/{result['total']}")


if __name__ == '__main__':
    main()
