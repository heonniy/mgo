"""Summarize only complete, unfiltered primary repeats from guarded jobs."""
import hashlib
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[1] / 'experiments/main_table_2x2_20261008'
WORKLOAD = Path('/home/hwlee/mgo-results/main_table_2x2_20261008')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SYSTEMS = ('ours', 'deepspeed', 'infinity', 'llama')
DISPLAY = {'ours': 'main_OURS', 'deepspeed': 'DeepSpeed ZeRO-Inference',
           'infinity': 'MoE-Infinity (repaired)', 'llama': 'llama.cpp balanced'}


def summarize(samples):
    return dict(median=statistics.median(samples), minimum=min(samples), maximum=max(samples),
                mean=statistics.mean(samples), stdev=statistics.stdev(samples), samples=samples)


def main():
    cases = []
    for dataset in ('ShareGPT', 'LMSYS-Chat-1M'):
        manifest_path = WORKLOAD / dataset / 'WORKLOADS.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['status'] == 'FROZEN' and len(manifest['cells']) == 8
        for cell in manifest['cells']:
            model = cell['model']
            for system in SYSTEMS:
                prefix = (f'mt2_{"qwen" if model == "Qwen3" else "deepseek"}_'
                          f'{dataset.lower().replace("-", "_")}_b{cell["local_batch"]}_'
                          f'l{cell["input_tokens"]}_{system}_r3')
                attempts = sorted(JOBS.glob(prefix + '_v*'))
                passed = []
                for path in attempts:
                    status_path = path / 'status.json'
                    if status_path.exists() and json.loads(status_path.read_text()).get('status') == 'PASS':
                        passed.append(path)
                case = dict(dataset=dataset, model=model, input_tokens=cell['input_tokens'],
                            local_batch=cell['local_batch'], global_batch=cell['global_requests'],
                            cache_percent=30, system=system, cell=cell['cell'],
                            attempts=[p.name for p in attempts], status='PENDING')
                if passed:
                    path = passed[-1]
                    status = json.loads((path / 'status.json').read_text())
                    cmd = status['command']
                    assert '--cell' in cmd and cmd[cmd.index('--cell') + 1] == cell['cell']
                    assert '--repeats' in cmd and cmd[cmd.index('--repeats') + 1] == '3'
                    if 'workload_manifest_sha256' in status:
                        assert status['workload_manifest'] == str(manifest_path)
                        assert status['workload_manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                    samples = [json.loads((path / f'repeat{i}.json').read_text()) for i in (1, 2, 3)]
                    assert all(row['status'] == 'PASS' for row in samples)
                    assert all(row['output_tokens'] == 64 and row['global_requests'] == cell['global_requests'] for row in samples)
                    expected_ids = [row['request_id'] for row in json.loads(Path(cell['target']['path']).read_text())['requests']]
                    if system in ('ours', 'deepspeed'):
                        for repeat in (1, 2, 3):
                            actual_ids = []
                            for rank in range(4):
                                row = json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                                actual_ids.extend(row['request_ids'])
                            assert actual_ids == expected_ids, (path.name, repeat, 'request membership/order mismatch')
                    else:
                        assert all(row['request_ids'] == expected_ids for row in samples)
                    case.update(status='PASS', selected_attempt=path.name, source_commit=status['source_commit'],
                                TTFT=summarize([row['TTFT'] for row in samples]),
                                TPOT=summarize([row['TPOT'] for row in samples]),
                                E2E=summarize([row['E2E'] for row in samples]))
                elif attempts:
                    latest = attempts[-1] / 'status.json'
                    if latest.exists():
                        case['status'] = json.loads(latest.read_text())['status']
                cases.append(case)
    assert len(cases) == 64
    completed = sum(row['status'] == 'PASS' for row in cases)
    output = dict(status='PASS' if completed == 64 else 'PARTIAL', completed=completed,
                  total=64, clean_repeats=3, primary_aggregate='median; full range and all samples retained',
                  cases=cases)
    (REPO / 'PROGRESS.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Two-model C30 main-table progress', '', f'Validated rows: **{completed}/64**.', '',
             'Each completed row has three unfiltered clean measurements. Times are seconds; TPOT is seconds per generated token and includes attention.', '',
             '| Dataset | Model | B/rank | Input | System | TTFT median [range] | TPOT median [range] | E2E median [range] | Status |',
             '|---|---|---:|---:|---|---:|---:|---:|---|']
    for case in cases:
        def value(key):
            if case['status'] != 'PASS':
                return '—'
            metric = case[key]
            return f'{metric["median"]:.3f} [{metric["minimum"]:.3f}, {metric["maximum"]:.3f}]'
        lines.append(f'| {case["dataset"]} | {case["model"]} | {case["local_batch"]} | '
                     f'{case["input_tokens"]} | {DISPLAY[case["system"]]} | {value("TTFT")} | '
                     f'{value("TPOT")} | {value("E2E")} | {case["status"]} |')
    (REPO / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
    print(f'{completed}/64 validated rows')


if __name__ == '__main__':
    main()
