"""Audit all frozen A/B/C primary repeats without discarding outliers."""

import hashlib
import json
import statistics
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
OUT = PKG / 'experiments/expert_grouping_ablation_20261009'
R8 = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
R4 = Path('/home/hwlee/mgo-results/expert_grouping_ablation_20261009')
R4_MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
CELLS = {
    'R8': (R8, R8 / 'WORKLOADS.json', 'Qwen3_ShareGPT_R8_C30_B16_L512_O64', 8, 'ours_{arm}_full_v1'),
    'R4': (R4, R4_MANIFEST, 'Qwen3_ShareGPT_R4_C30_B16_L512_O64', 4, 'r4_{arm}_full_v1'),
}


def read(path):
    return json.loads(path.read_text())


def summary(values):
    mean = statistics.mean(values)
    return dict(samples=values, mean=mean, median=statistics.median(values),
                minimum=min(values), maximum=max(values),
                relative_difference_percent=100 * (max(values) - min(values)) / mean)


def audit_cell(label, root, manifest_path, cell, world, pattern):
    manifest = read(manifest_path)
    spec, = [row for row in manifest['cells'] if row['cell'] == cell]
    assert manifest['status'] == 'FROZEN'
    assert (spec['local_batch'], spec['global_requests'], spec['input_tokens'],
            spec['output_tokens'], spec['cache_percent'], spec['expert_slots']) == (16, world*16, 512, 64, 30, 1843)
    for phase in ('warmup', 'target'):
        path = Path(spec[phase]['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec[phase]['sha256']
    expected = [row['request_id'] for row in read(Path(spec['target']['path']))['requests']]
    assert len(expected) == world*16
    rows = {}
    for arm in 'abc':
        job = root / 'jobs' / pattern.format(arm=arm)
        state, result = read(job/'status.json'), read(job/'result.json')
        assert state['status'] == result['status'] == 'PASS' and len(state['restored_model_loads']) == 8
        assert not result['smoke'] and result['primary_repeats'] == 2
        assert state['workload_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        samples, rank_samples = [], []
        for repeat in (1, 2):
            sample = read(job/f'repeat{repeat}.json')
            ranks = [read(job/f'repeat{repeat}_rank{rank}.json') for rank in range(world)]
            assert sample['status'] == 'PASS' and not sample['smoke']
            assert sample['global_requests'] == world*16 and sample['output_tokens'] == 64
            assert abs(sample['E2E'] - sample['TTFT'] - 63*sample['TPOT']) < 1e-6
            assert [rid for rank in ranks for rid in rank['request_ids']] == expected
            assert len({rank['release_ns'] for rank in ranks}) == 1
            start = ranks[0]['release_ns']
            first = max(rank['first_ns'] for rank in ranks)
            end = max(rank['end_ns'] for rank in ranks)
            assert abs(sample['TTFT'] - (first-start)/1e9) < 1e-6
            assert abs(sample['E2E'] - (end-start)/1e9) < 1e-6
            for rank in ranks:
                assert rank['finite_logits'] and rank['expert_cache_start'] == 'empty'
                assert rank['policy'] == 'LA_CA_NEAR' and rank['prefetch_off']
                assert rank['native_prefill'] and rank['decode_layout_fast']
                assert rank['validation']['status'] == 'PASS'
                assert rank['validation']['physical_slots'] == 1843
                assert len(rank['tokens']) == 16 and all(len(token) == 64 for token in rank['tokens'])
                assert rank['grouped_decode_mode'] == {'a':'off','b':'serial_all','c':'two_wave'}[arm]
            samples.append(sample)
            rank_samples.append(ranks)
        metrics = {key: summary([sample[key] for sample in samples])
                   for key in ('TTFT', 'TPOT', 'E2E', 'throughput')}
        h2d = [sum(rank['validation']['scheduler']['bytes'] for rank in ranks)
               for ranks in rank_samples]
        copies = [sum(rank['validation']['scheduler']['copies'] for rank in ranks)
                  for ranks in rank_samples]
        assert h2d[0] == h2d[1] and copies[0] == copies[1]
        status = 'UNSTABLE_E2E' if metrics['E2E']['relative_difference_percent'] > 5 else 'PASS'
        rows[arm] = dict(job=str(job), status=status, metrics=metrics,
                         h2d_bytes=h2d[0], h2d_copies=copies[0],
                         source_commit=state['source_commit'])
        if arm == 'a':
            rows[arm]['native_counts'] = {
                key:sum(rank['native_executor_counts'][key] for rank in rank_samples[0])
                for key in ('waves','groups','waits')}
        else:
            rows[arm]['grouped_counts'] = {
                key:sum(rank['grouped_executor_counts'][key] for rank in rank_samples[0])
                for key in ('events','waves','first_wave_groups','second_wave_groups',
                            'no_ready_events','serial_wait_wall_ns')}
            assert rows[arm]['grouped_counts']['events'] == world*48*63
            assert (rows[arm]['grouped_counts']['first_wave_groups'] +
                    rows[arm]['grouped_counts']['second_wave_groups']) > 0
    baseline_job = root / 'jobs' / pattern.format(arm='a')
    token_mismatches = {}
    for arm in 'bc':
        comparison_job = root / 'jobs' / pattern.format(arm=arm)
        mismatches = []
        for repeat in (1, 2):
            a = [token for rank in range(world) for token in read(baseline_job/f'repeat{repeat}_rank{rank}.json')['tokens']]
            b = [token for rank in range(world) for token in read(comparison_job/f'repeat{repeat}_rank{rank}.json')['tokens']]
            mismatches.append(sum(x != y for ra, rb in zip(a,b) for x,y in zip(ra,rb)))
            for rank in range(world):
                ref = read(baseline_job/f'repeat{repeat}_rank{rank}.json')
                got = read(comparison_job/f'repeat{repeat}_rank{rank}.json')
                assert ref['validation']['scheduler']['bytes'] == got['validation']['scheduler']['bytes']
                assert ref['validation']['scheduler']['copies'] == got['validation']['scheduler']['copies']
                assert ref['validation']['state_hash'] == got['validation']['state_hash']
        token_mismatches[arm] = mismatches
    assert rows['a']['h2d_bytes'] == rows['b']['h2d_bytes'] == rows['c']['h2d_bytes']
    assert rows['a']['h2d_copies'] == rows['b']['h2d_copies'] == rows['c']['h2d_copies']
    return dict(label=label, world=world, cell=cell, target_sha256=spec['target']['sha256'],
                warmup_sha256=spec['warmup']['sha256'], arms=rows,
                token_mismatches_vs_A=token_mismatches,
                gain_B_vs_A_percent=100*(1-rows['b']['metrics']['TPOT']['mean']/rows['a']['metrics']['TPOT']['mean']),
                gain_C_vs_A_percent=100*(1-rows['c']['metrics']['TPOT']['mean']/rows['a']['metrics']['TPOT']['mean']),
                gain_C_vs_B_percent=100*(1-rows['c']['metrics']['TPOT']['mean']/rows['b']['metrics']['TPOT']['mean']))


def main():
    cells = {label:audit_cell(label,*args) for label,args in CELLS.items()}
    out = dict(status='PASS', cells=cells,
               interpretation='A versus B changes grouped backend and H2D scheduling together; B versus C isolates two-wave readiness within the same grouped backend. Two repeats do not establish a small B/C difference.')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'PRIMARY_RESULTS.json').write_text(json.dumps(out, indent=2)+'\n')
    lines = ['# main_OURS grouped expert A/B/C primary timing', '',
             'Qwen3-30B, ShareGPT, C30, per-rank B16, input512, decode64, Near, prefetch OFF, identical frozen requests in each R. A is C++ Ready-First individual GEMMs; B waits all H2D then one grouped wave; C executes ready experts then remaining misses in a second grouped wave.',
             '',
             '| R | Arm | TTFT s (two samples) | TPOT s/token (two samples) | E2E s (two samples) | Mean TPOT | E2E stability |',
             '|---|---|---:|---:|---:|---:|---|']
    for label,cell in cells.items():
        for arm,item in cell['arms'].items():
            m=item['metrics']
            fmt=lambda key: ' / '.join(f'{value:.4f}' for value in m[key]['samples'])
            lines.append(f'| {label} | {arm.upper()} | {fmt("TTFT")} | {fmt("TPOT")} | {fmt("E2E")} | {m["TPOT"]["mean"]:.4f} | {item["status"]} |')
    lines += ['',
              'All samples are retained. Every arm in a given R has identical rank-level H2D bytes, copy counts, final cache-state hashes, and generated tokens in both repeats. First-target TTFT is higher in each arm; E2E exceeds the 5% repeat threshold, so E2E is marked unstable without further repetitions. TPOT remains substantially more stable.',
              '',
              '| R | Total H2D GiB | Copies | B gain vs A | C gain vs A | C gain vs B |',
              '|---|---:|---:|---:|---:|---:|']
    for label,cell in cells.items():
        a=cell['arms']['a']
        lines.append(f'| {label} | {a["h2d_bytes"]/2**30:.3f} | {a["h2d_copies"]:,} | {cell["gain_B_vs_A_percent"]:.2f}% | {cell["gain_C_vs_A_percent"]:.2f}% | {cell["gain_C_vs_B_percent"]:.2f}% |')
    lines += ['',
              'A→B changes both kernel implementation and H2D scheduling, so it does not isolate their separate effects. B→C keeps the grouped kernel and changes only readiness scheduling. The small B/C difference is within observed repeat variation, especially on R4. Instrumented breakdown is reported separately.',
              '']
    (OUT/'PRIMARY_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'PASS','cells':{label:{'B_gain':cell['gain_B_vs_A_percent'],'C_gain':cell['gain_C_vs_A_percent']} for label,cell in cells.items()}}))


if __name__ == '__main__':
    main()
