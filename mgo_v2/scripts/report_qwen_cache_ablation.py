"""Validate and summarize guarded Qwen main_OURS cache-capacity runs."""

import csv
import hashlib
import json
import math
import statistics
from itertools import combinations
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/qwen_cache_ablation_20261009'
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PERCENTS = (20, 30, 40, 50)
GPUS = (0, 1, 4, 5)
EXPERT_BYTES = 9 * 2**20
PINNED_BYTES = 48 * 128 * EXPERT_BYTES


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def summary(values):
    return dict(samples=values, median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                mean=statistics.mean(values), stdev=statistics.stdev(values))


def attempt_number(path):
    return int(path.name.rsplit('_v', 1)[1])


def verify(path, cell, expected_ids, manifest_hash):
    status = read(path / 'status.json')
    assert status['status'] == 'PASS'
    assert status['workload_manifest_sha256'] == manifest_hash
    command = status['command']
    assert command[command.index('--cell') + 1] == cell['cell']
    assert command[command.index('--repeats') + 1] == '3' and '--smoke' not in command
    for flag in ('--expert-executor', '--native-prefill', '--prefetch-off',
                 '--prefill-optimized', '--prefill-layout-fast', '--decode-layout-fast',
                 '--policy'):
        assert flag in command
    assert command[command.index('--expert-executor') + 1] == 'native'
    assert command[command.index('--policy') + 1] == 'LA_CA_NEAR'
    assert '--post-generation-diagnostic' not in command
    metrics = {key: [] for key in ('TTFT', 'TPOT', 'E2E')}
    h2d = [[] for _ in GPUS]
    tokens = {}
    for repeat in range(4):
        batch = read(path / f'repeat{repeat}.json')
        assert batch['status'] == 'PASS'
        assert batch['output_tokens'] == 64 and batch['global_requests'] == 64
        assert math.isclose(batch['E2E'], batch['TTFT'] + 63 * batch['TPOT'], abs_tol=1e-6)
        phase = 'warmup' if repeat == 0 else 'target'
        requests, token_lists = [], []
        for rank in range(4):
            item = read(path / f'repeat{repeat}_rank{rank}.json')
            assert item['rank'] == rank and item['physical_gpu'] == GPUS[rank]
            assert item['expert_executor'] == 'native' and item['native_prefill']
            assert item['prefetch_off'] and item['prefill_optimized']
            assert item['prefill_layout_fast'] and item['decode_layout_fast']
            assert item['policy'] == 'LA_CA_NEAR' and item['expert_cache_start'] == 'empty'
            assert item['pinned_host_bytes'] == PINNED_BYTES
            assert item['validation']['status'] == 'PASS' and item['no_compile']
            assert item['validation']['main_slots'] == cell['expert_slots'] - 8
            assert item['validation']['physical_slots'] == cell['expert_slots']
            scheduler = item['validation']['scheduler']
            assert scheduler['background_copies'] == 0
            assert scheduler['bytes'] == scheduler['copies'] * EXPERT_BYTES
            assert len(item['tokens']) == 16 and all(len(row) == 64 for row in item['tokens'])
            requests.extend(item['request_ids'])
            token_lists.extend(item['tokens'])
            if repeat:
                h2d[rank].append(scheduler['bytes'])
        assert requests == expected_ids[phase]
        if repeat:
            tokens[repeat] = token_lists
            for key in metrics:
                metrics[key].append(batch[key])
    for rank in range(4):
        prefill = read(path / f'prefill_validation_rank{rank}.json')
        assert prefill['status'] == 'PASS'
        assert prefill['metadata_exact_checks'] == 48
        assert prefill['layout_exact_checks'] == 48
    sampled_peak = {gpu: 0 for gpu in GPUS}
    host_min = None
    for line in (path / 'resources.jsonl').read_text().splitlines():
        item = json.loads(line)
        assert sorted(g['gpu'] for g in item['gpus']) == list(GPUS)
        host_min = item['host_available'] if host_min is None else min(host_min, item['host_available'])
        for gpu in item['gpus']:
            sampled_peak[gpu['gpu']] = max(sampled_peak[gpu['gpu']], gpu['used_mib'])
    assert host_min and all(sampled_peak.values())
    return dict(status='PASS', selected_attempt=path.name,
                source_commit=status['source_commit'],
                metrics={name: summary(values) for name, values in metrics.items()},
                h2d_bytes_by_rank=[summary(values) for values in h2d],
                sampled_peak_hbm_mib_by_physical_gpu=sampled_peak,
                min_host_available_bytes=host_min,
                full_output_agreement_1_2=sum(a == b for a, b in zip(tokens[1], tokens[2])),
                full_output_agreement_1_3=sum(a == b for a, b in zip(tokens[1], tokens[3])),
                full_output_agreement_2_3=sum(a == b for a, b in zip(tokens[2], tokens[3])))


def main():
    manifest = read(MANIFEST)
    assert manifest['status'] == 'FROZEN' and manifest['dataset'] == 'ShareGPT'
    assert manifest['physical_gpus'] == list(GPUS) and len(manifest['cells']) == 4
    assert sorted(c['cache_percent'] for c in manifest['cells']) == list(PERCENTS)
    same_inputs = {'warmup': [], 'target': []}
    cases = []
    for cell in sorted(manifest['cells'], key=lambda c: c['cache_percent']):
        percent = cell['cache_percent']
        slots = 48 * 128 * percent // 100
        per_rank = [slots // 4 + (rank < slots % 4) for rank in range(4)]
        assert cell['model'] == 'Qwen3' and cell['local_batch'] == 16
        assert cell['input_tokens'] == 512 and cell['output_tokens'] == 64
        assert cell['expert_slots'] == slots and cell['expert_slots_per_rank'] == per_rank
        assert cell['expert_budget_bytes'] == slots * EXPERT_BYTES
        expected_ids = {}
        for phase in ('warmup', 'target'):
            source = Path(cell[phase]['path'])
            assert digest(source) == cell[phase]['sha256']
            rows = read(source)['requests']
            assert len(rows) == 64 and all(len(row['input_ids']) == 512 for row in rows)
            expected_ids[phase] = [row['request_id'] for row in rows]
            same_inputs[phase].append(rows)
        assert set(expected_ids['warmup']).isdisjoint(expected_ids['target'])
        prefix = f'qca_c{percent}_b16_ours_r3_v'
        attempts = sorted(JOBS.glob(prefix + '*'), key=attempt_number)
        passed = [path for path in attempts if (path / 'status.json').exists()
                  and read(path / 'status.json').get('status') == 'PASS']
        case = dict(cell=cell['cell'], model='Qwen3', dataset='ShareGPT',
                    cache_percent=percent, expert_slots=slots,
                    expert_slots_per_rank=per_rank, local_batch=16,
                    input_tokens=512, output_tokens=64,
                    attempts=[path.name for path in attempts], status='PENDING')
        if passed:
            case.update(verify(passed[-1], cell, expected_ids, digest(MANIFEST)))
        elif attempts:
            last = attempts[-1] / 'status.json'
            case['status'] = read(last).get('status', 'RUNNING') if last.exists() else 'RUNNING'
        cases.append(case)
    assert all(inputs == same_inputs['warmup'][0] for inputs in same_inputs['warmup'])
    assert all(inputs == same_inputs['target'][0] for inputs in same_inputs['target'])
    completed = sum(c['status'] == 'PASS' for c in cases)
    cross_capacity_agreement = []
    for left, right in combinations((c for c in cases if c['status'] == 'PASS'), 2):
        def output_tokens(case):
            root = JOBS / case['selected_attempt']
            return [tokens for rank in range(4)
                    for tokens in read(root / f'repeat2_rank{rank}.json')['tokens']]
        left_tokens, right_tokens = output_tokens(left), output_tokens(right)
        assert len(left_tokens) == len(right_tokens) == 64
        cross_capacity_agreement.append(dict(
            left_cache_percent=left['cache_percent'],
            right_cache_percent=right['cache_percent'],
            same_full_output_requests=sum(a == b for a, b in zip(left_tokens, right_tokens)),
            same_first_token_requests=sum(a[0] == b[0] for a, b in zip(left_tokens, right_tokens)),
            same_token_positions=sum(x == y for a, b in zip(left_tokens, right_tokens)
                                     for x, y in zip(a, b))))
    output = dict(status='PASS' if completed == 4 else 'PARTIAL', completed=completed,
                  total=4, clean_target_repeats=3, source_manifest_sha256=digest(MANIFEST),
                  cases=cases, cross_capacity_agreement=cross_capacity_agreement)
    write(REPORT / 'PROGRESS.json', output)
    lines = ['# Qwen3 ShareGPT main_OURS cache-capacity sweep', '',
             f'Validated cells: **{completed}/4**. R4 GPUs 0/1/4/5, B16/rank, input512/output64. '
             'Three unfiltered target repeats; figures are median [minimum, maximum] in seconds, '
             'with TPOT in seconds per token.', '',
             '| Cache | TTFT | TPOT | E2E | H2D GiB, four ranks | Status |',
             '|---:|---:|---:|---:|---:|---|']
    with (REPORT / 'RESULTS.csv').open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['cache_percent', 'status', 'TTFT_median_s', 'TTFT_min_s', 'TTFT_max_s',
                         'TPOT_median_s_per_token', 'TPOT_min_s_per_token', 'TPOT_max_s_per_token',
                         'E2E_median_s', 'E2E_min_s', 'E2E_max_s', 'H2D_median_GiB_all_ranks',
                         'expert_slots', 'selected_attempt'])
        for case in cases:
            def display(key, scale=1):
                if case['status'] != 'PASS':
                    return '—'
                item = case['metrics'][key] if key in case.get('metrics', {}) else None
                if item is None:
                    vals = [r['median'] for r in case['h2d_bytes_by_rank']]
                    return f'{sum(vals) / scale:.3f}'
                return f'{item["median"] / scale:.3f} [{item["minimum"] / scale:.3f}, {item["maximum"] / scale:.3f}]'
            lines.append(f'| {case["cache_percent"]}% | {display("TTFT")} | '
                         f'{display("TPOT")} | {display("E2E")} | '
                         f'{display("H2D", 2**30)} | {case["status"]} |')
            values = [case['metrics'][key][stat] if case['status'] == 'PASS' else ''
                      for key in ('TTFT', 'TPOT', 'E2E')
                      for stat in ('median', 'minimum', 'maximum')]
            h2d_gib = (sum(r['median'] for r in case['h2d_bytes_by_rank']) / 2**30
                       if case['status'] == 'PASS' else '')
            writer.writerow([case['cache_percent'], case['status'], *values, h2d_gib,
                             case['expert_slots'], case.get('selected_attempt', '')])
    lines.extend(['', 'main_OURS uses Near/native expert execution, compiled prefill/decode '
                  'layout, full-pinned host expert source, and prefetch OFF. Raw prompts and '
                  'token IDs remain outside Git. H2D GiB is the sum of four rank counters '
                  'per measured batch, including prefill.'])
    (REPORT / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
    token_lines = ['# Output token stability', '',
                   'Full 64-token request agreement across the three target repeats '
                   '(out of 64 requests). Raw IDs remain outside Git.', '',
                   '| Cache | 1 versus 2 | 1 versus 3 | 2 versus 3 |',
                   '|---:|---:|---:|---:|']
    for case in cases:
        if case['status'] == 'PASS':
            fields = [case[f'full_output_agreement_{pair}'] for pair in ('1_2', '1_3', '2_3')]
            token_lines.append(f'| {case["cache_percent"]}% | '
                               + ' | '.join(f'{value}/64' for value in fields) + ' |')
        else:
            token_lines.append(f'| {case["cache_percent"]}% | — | — | — |')
    token_lines.extend(['', 'Across capacities, identical input requests can lead to different '
                        'autoregressive output paths. Compare the second target repeat in '
                        'each passed cell; these are aggregate counts, not raw token IDs.', '',
                        '| Cache pair | Same first token | Same complete 64-token sequence | '
                        'Same token positions |', '|---|---:|---:|---:|'])
    for pair in cross_capacity_agreement:
        token_lines.append(f'| C{pair["left_cache_percent"]}–C{pair["right_cache_percent"]} | '
                           f'{pair["same_first_token_requests"]}/64 | '
                           f'{pair["same_full_output_requests"]}/64 | '
                           f'{pair["same_token_positions"]}/4096 |')
    (REPORT / 'TOKEN_STABILITY.md').write_text('\n'.join(token_lines) + '\n')
    print(f'{completed}/4 validated Qwen cache cells')


if __name__ == '__main__':
    main()
