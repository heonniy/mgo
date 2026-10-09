"""Validate and summarize guarded DeepSeek cache-sweep measurements."""

import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from report_deepseek_cache_token_stability import main as report_token_stability


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/deepseek_cache_ablation_20261009'
MANIFEST = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SYSTEMS = ('ours', 'infinity', 'deepspeed', 'llama')
LABELS = {'ours': 'main_OURS', 'infinity': 'MoE-Infinity (repaired)',
          'deepspeed': 'DeepSpeed ZeRO-Inference', 'llama': 'llama.cpp balanced'}
EXPERT_BYTES = 3 * 2048 * 1408 * 2
PHYSICAL_GPUS = (0, 1, 4, 5)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def requests(cell):
    ids = {}
    for phase in ('warmup', 'target'):
        path = Path(cell[phase]['path'])
        assert digest(path) == cell[phase]['sha256']
        rows = read(path)['requests']
        assert len(rows) == cell['global_requests']
        assert all(len(row['input_ids']) == 512 for row in rows)
        ids[phase] = [row['request_id'] for row in rows]
        assert len(ids[phase]) == len(set(ids[phase]))
    assert set(ids['warmup']).isdisjoint(ids['target'])
    return ids


def summarize(values):
    return dict(samples=values, median=statistics.median(values),
                mean=statistics.mean(values), stdev=statistics.stdev(values),
                minimum=min(values), maximum=max(values))


def attempt_number(path):
    return int(path.name.rsplit('_v', 1)[1])


def verify_attempt(path, cell, system, expected_ids, manifest_digest):
    status = read(path / 'status.json')
    assert status['status'] == 'PASS'
    assert status['workload_manifest_sha256'] == manifest_digest
    command = status['command']
    assert command[command.index('--cell') + 1] == cell['cell']
    assert command[command.index('--repeats') + 1] == '3'
    assert '--smoke' not in command
    samples = [read(path / f'repeat{index}.json') for index in range(4)]
    budget = [slot * EXPERT_BYTES for slot in cell['expert_slots_per_rank']]
    for repeat, row in enumerate(samples):
        assert row['status'] == 'PASS'
        assert row['output_tokens'] == 64 and row['global_requests'] == cell['global_requests']
        assert math.isclose(row['E2E'], row['TTFT'] + 63 * row['TPOT'], abs_tol=1e-6)
        phase = 'warmup' if repeat == 0 else 'target'
        if system in ('ours', 'deepspeed'):
            rank_rows = [read(path / f'repeat{repeat}_rank{rank}.json') for rank in range(4)]
            actual = list(itertools.chain.from_iterable(r['request_ids'] for r in rank_rows))
            assert actual == expected_ids[phase]
            for rank, ranked in enumerate(rank_rows):
                if system == 'ours':
                    assert ranked['policy'] == 'LA_CA_NEAR' and ranked['prefetch_off'] is True
                    assert ranked['expert_executor'] == 'native_deepseek'
                    assert ranked['cache_start'] == 'empty'
                    assert ranked['physical_cache_slots'] == cell['expert_slots_per_rank'][rank]
                    assert ranked['cache_capacity_slots'] == cell['expert_slots_per_rank'][rank] - 2
                else:
                    assert ranked['all_parameter_peak_bytes'] <= ranked['parameter_budget_bytes']
                    assert ranked['parameter_budget_bytes'] == cell['expert_budget_bytes'] // 4
        else:
            assert row['request_ids'] == expected_ids[phase]
            if system == 'infinity':
                assert row['expert_budget_per_gpu'] == budget
                assert row['cache_before']['resident_count'] == 0
                assert row['eam_calls'] == 26 * 64
                assert row['eam_candidates'] == 0
                assert all(row['cache_after'][f'gpu_{rank}_peak_charged_bytes'] <= budget[rank]
                           for rank in range(4))
            else:
                layers_per_rank = min(cell['expert_slots_per_rank']) // 64
                assert row['expert_placement'] == f'balanced{4 * layers_per_rank}'
                assert row['gpu_expert_layers_by_device'] == {
                    f'CUDA{rank}': layers_per_rank for rank in range(4)}
                assert row['expert_resident_bytes'] == 4 * layers_per_rank * 64 * EXPERT_BYTES
                assert row['synchronous_batch'] and row['cpu_threads'] == 32
                assert row['finite_logits'] and row['cache_start'] == 'empty KV; static weight partition retained'
    if system == 'llama':
        placement = read(path / 'placement_audit.json')
        config = read(path / 'config.json')
        assert placement['status'] == 'PASS' and placement['op_offload'] is False
        assert config['cuda_graphs_runtime'] == 'off' and config['llama_graph_reuse'] is False
        assert config['synchronous_batch'] is True
        layers_per_rank = min(cell['expert_slots_per_rank']) // 64
        assert all(layers_per_rank * 64 * EXPERT_BYTES <= value for value in budget)
    if system == 'infinity':
        backend = read(path / 'attention_backend.json')
        assert backend['status'] == 'PASS' and backend['backend'] == 'eager'
    sampled_hbm = {gpu: 0 for gpu in PHYSICAL_GPUS}
    for line in (path / 'resources.jsonl').read_text().splitlines():
        for gpu in json.loads(line)['gpus']:
            if gpu['gpu'] in sampled_hbm:
                sampled_hbm[gpu['gpu']] = max(sampled_hbm[gpu['gpu']], gpu['used_mib'])
    assert all(sampled_hbm.values())
    result = dict(status='PASS', selected_attempt=path.name,
                  source_commit=status['source_commit'],
                  sampled_peak_hbm_mib_by_physical_gpu=sampled_hbm)
    for metric in ('TTFT', 'TPOT', 'E2E'):
        result[metric] = summarize([row[metric] for row in samples[1:]])
    if system == 'ours':
        for key in ('h2d_bytes', 'remote_dispatch_bytes', 'remote_return_bytes'):
            result[key] = summarize([
                sum(read(path / f'repeat{repeat}_rank{rank}.json')[key] for rank in range(4))
                for repeat in (1, 2, 3)])
    if system == 'infinity':
        result['evictions'] = summarize([
            row['cache_after']['evictions'] for row in samples[1:]])
    return result


def main():
    manifest = read(MANIFEST)
    assert manifest['status'] == 'FROZEN' and manifest['dataset'] == 'ShareGPT'
    assert manifest['physical_gpus'] == list(PHYSICAL_GPUS)
    batches = sorted({cell['local_batch'] for cell in manifest['cells']})
    assert batches and set(batches).issubset({16, 64})
    expected = set(itertools.product(batches, (20, 30, 40, 50)))
    actual = {(cell['local_batch'], cell['cache_percent']) for cell in manifest['cells']}
    assert len(manifest['cells']) == len(actual) == len(expected) and actual == expected
    cases = []
    for cell in sorted(manifest['cells'], key=lambda c: (c['local_batch'], c['cache_percent'])):
        expected_ids = requests(cell)
        percent, batch = cell['cache_percent'], cell['local_batch']
        slots = 26 * 64 * percent // 100
        assert cell['expert_slots'] == slots
        assert cell['expert_slots_per_rank'] == [slots // 4 + (rank < slots % 4)
                                                  for rank in range(4)]
        assert cell['expert_budget_bytes'] == slots * EXPERT_BYTES
        for system in SYSTEMS:
            prefix = f'dca_c{percent}_b{batch}_{system}_r3_v'
            attempts = sorted(JOBS.glob(prefix + '*'), key=attempt_number)
            passed = [path for path in attempts if (path / 'status.json').exists()
                      and read(path / 'status.json').get('status') == 'PASS']
            case = dict(dataset='ShareGPT', model='DeepSeekV2Lite', cache_percent=percent,
                        local_batch=batch, global_batch=4 * batch, input_tokens=512,
                        output_tokens=64, system=system, cell=cell['cell'],
                        expert_slots=slots, expert_slots_per_rank=cell['expert_slots_per_rank'],
                        expert_budget_bytes_per_rank=[x * EXPERT_BYTES for x in cell['expert_slots_per_rank']],
                        attempts=[path.name for path in attempts], status='PENDING')
            if passed:
                case.update(verify_attempt(passed[-1], cell, system, expected_ids, digest(MANIFEST)))
            elif attempts:
                latest = read(attempts[-1] / 'status.json') if (attempts[-1] / 'status.json').exists() else {}
                case['status'] = latest.get('status', 'RUNNING')
            cases.append(case)
    completed = sum(case['status'] == 'PASS' for case in cases)
    total = len(expected) * len(SYSTEMS)
    output = dict(status='PASS' if completed == total else 'PARTIAL', completed=completed,
                  total=total, clean_target_repeats=3, source_manifest_sha256=digest(MANIFEST),
                  cases=cases)
    write(REPORT / 'PROGRESS.json', output)
    lines = ['# DeepSeek ShareGPT cache-capacity ablation', '',
             f'Validated system cells: **{completed}/{total}**. Input 512, output 64, R4 GPUs 0/1/4/5. '
             'Three unfiltered target repeats per completed row; values below are median [minimum, maximum] seconds, '
             'with TPOT in seconds per generated token.', '',
             '| B/rank | Cache | System | TTFT | TPOT | E2E | Status |',
             '|---:|---:|---|---:|---:|---:|---|']
    csv_path = REPORT / 'RESULTS.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['batch_per_rank', 'cache_percent', 'system', 'status', 'TTFT_median_s',
                         'TTFT_min_s', 'TTFT_max_s', 'TPOT_median_s_per_token', 'TPOT_min_s_per_token',
                         'TPOT_max_s_per_token', 'E2E_median_s', 'E2E_min_s', 'E2E_max_s',
                         'expert_slots', 'selected_attempt'])
        for case in cases:
            def metric(name):
                if case['status'] != 'PASS':
                    return '—'
                item = case[name]
                return f'{item["median"]:.3f} [{item["minimum"]:.3f}, {item["maximum"]:.3f}]'
            lines.append(f'| {case["local_batch"]} | {case["cache_percent"]}% | '
                         f'{LABELS[case["system"]]} | {metric("TTFT")} | {metric("TPOT")} | '
                         f'{metric("E2E")} | {case["status"]} |')
            values = [case[key][stat] if case['status'] == 'PASS' else ''
                      for key in ('TTFT', 'TPOT', 'E2E')
                      for stat in ('median', 'minimum', 'maximum')]
            writer.writerow([case['local_batch'], case['cache_percent'], LABELS[case['system']],
                             case['status'], *values, case['expert_slots'],
                             case.get('selected_attempt', '')])
    lines.extend(['', 'main_OURS uses Near/native experts with prefetch OFF. DeepSeek MoE-Infinity '
                  'uses EAM eviction priorities with speculative admission OFF. llama.cpp uses whole-layer '
                  'balanced placement: 1/1/1/1 GPU expert layers at C20/C30, 2/2/2/2 at C40, '
                  'and 3/3/3/3 at C50. C30 limits expert residency, not total HBM.',
                  '', 'Raw frozen token-ID manifests and request lists remain outside Git.'])
    (REPORT / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
    activity = ['# Cache activity', '',
                'These are separate implementation counters, not cross-system-equivalent miss counts. '
                'main_OURS H2D and peer bytes are sums of all four ranks per complete measured batch; '
                'MoE-Infinity evictions come from its EAM cache. Values are medians of the three '
                'unfiltered target repeats with full ranges.', '',
                '| B/rank | Cache | System | H2D GiB | Dispatch GiB | Return GiB | EAM evictions |',
                '|---:|---:|---|---:|---:|---:|---:|']
    for case in cases:
        if case['system'] not in ('ours', 'infinity'):
            continue
        def activity_value(key, scale=1):
            if case['status'] != 'PASS' or key not in case:
                return '—'
            row = case[key]
            return (f'{row["median"] / scale:.3f} '
                    f'[{row["minimum"] / scale:.3f}, {row["maximum"] / scale:.3f}]')
        activity.append(f'| {case["local_batch"]} | {case["cache_percent"]}% | '
                        f'{LABELS[case["system"]]} | {activity_value("h2d_bytes", 2**30)} | '
                        f'{activity_value("remote_dispatch_bytes", 2**30)} | '
                        f'{activity_value("remote_return_bytes", 2**30)} | '
                        f'{activity_value("evictions")} |')
    (REPORT / 'CACHE_ACTIVITY.md').write_text('\n'.join(activity) + '\n')
    report_token_stability()
    print(f'{completed}/{total} validated cache-ablation rows')


if __name__ == '__main__':
    main()
