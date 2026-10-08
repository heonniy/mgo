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


def memory_receipt(path, cell, system, samples):
    expert_bytes = cell['expert_budget_bytes'] // cell['expert_slots']
    assert expert_bytes * cell['expert_slots'] == cell['expert_budget_bytes']
    expert_budget = [slots * expert_bytes for slots in cell['expert_slots_per_rank']]
    sampled_hbm = {gpu: 0 for gpu in (0, 1, 4, 5)}
    with (path / 'resources.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            for gpu in record['gpus']:
                if gpu['gpu'] in sampled_hbm:
                    sampled_hbm[gpu['gpu']] = max(sampled_hbm[gpu['gpu']], gpu['used_mib'])
    assert all(sampled_hbm.values()), path
    allocated = [0] * 4
    reserved = [0] * 4
    pinned = [0] * 4
    infinity_peak_charged = [0] * 4
    if system in ('ours', 'deepspeed'):
        for repeat in (1, 2, 3):
            for rank in range(4):
                row = json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                allocated[rank] = max(allocated[rank], row['peak_allocated_bytes'])
                reserved[rank] = max(reserved[rank], row.get('peak_reserved_bytes', 0))
                pinned[rank] = max(pinned[rank], row['pinned_host_bytes'])
    elif system == 'infinity':
        for row in samples:
            for rank in range(4):
                allocated[rank] = max(allocated[rank], row['peak_allocated_bytes'][rank])
                reserved[rank] = max(reserved[rank], row['peak_reserved_bytes'][rank])
                charged = row['cache_after'][f'gpu_{rank}_peak_charged_bytes']
                assert charged <= expert_budget[rank], (path, rank, charged, expert_budget[rank])
                infinity_peak_charged[rank] = max(infinity_peak_charged[rank], charged)
            assert row['expert_budget_per_gpu'] == expert_budget
    elif system == 'llama':
        assert all(row['expert_resident_bytes'] <= sum(expert_budget) for row in samples)
    else:
        raise AssertionError(system)
    return dict(expert_budget_bytes_per_gpu=expert_budget,
                sampled_peak_hbm_mib_by_physical_gpu=sampled_hbm,
                torch_peak_allocated_bytes_by_rank=allocated if any(allocated) else None,
                torch_peak_reserved_bytes_by_rank=reserved if any(reserved) else None,
                pinned_host_bytes_by_rank=pinned if any(pinned) else None,
                infinity_peak_expert_charged_bytes_per_gpu=infinity_peak_charged if any(infinity_peak_charged) else None,
                llama_expert_resident_bytes=samples[0]['expert_resident_bytes'] if system == 'llama' else None)


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
                    if system == 'infinity':
                        expected_calls = (48 if model == 'Qwen3' else 26) * 64
                        assert all(row['eam_calls'] == expected_calls for row in samples)
                        if model != 'Qwen3':
                            policy = json.loads((path / 'eam_policy.json').read_text())
                            capacity_slots = sum(policy['expert_budget_per_gpu']) // policy['expert_bytes']
                            assert policy['min_free_expert_slots'] > capacity_slots
                            assert all(row['eam_candidates'] == 0 for row in samples)
                            case['baseline_adaptation'] = 'EAM eviction priorities on; speculative prefetch off after native expert-wait stall'
                    expected_ids = [row['request_id'] for row in json.loads(Path(cell['target']['path']).read_text())['requests']]
                    if system in ('ours', 'deepspeed'):
                        for repeat in (1, 2, 3):
                            actual_ids = []
                            for rank in range(4):
                                row = json.loads((path / f'repeat{repeat}_rank{rank}.json').read_text())
                                actual_ids.extend(row['request_ids'])
                                if system == 'ours':
                                    assert row['policy'] == 'LA_CA_NEAR' and row['prefetch_off'] is True
                                    assert row['expert_executor'] == (
                                        'native' if model == 'Qwen3' else 'native_deepseek')
                                    if model == 'Qwen3':
                                        assert all(row[key] is True for key in (
                                            'native_prefill', 'prefill_optimized',
                                            'prefill_layout_fast', 'decode_layout_fast'))
                                if system == 'deepspeed':
                                    assert 0 < row['all_parameter_peak_bytes'] <= row['parameter_budget_bytes']
                                    if model != 'Qwen3':
                                        assert row['parameter_counter_full_scans'] > 0
                                        assert row['parameter_counter_corrections'] >= 0
                            assert actual_ids == expected_ids, (path.name, repeat, 'request membership/order mismatch')
                    else:
                        assert all(row['request_ids'] == expected_ids for row in samples)
                        if system == 'llama':
                            placement = json.loads((path / 'placement_audit.json').read_text())
                            balanced = 3 if model == 'Qwen3' else 1
                            assert placement['status'] == 'PASS'
                            assert placement['expert_placement'] == (
                                'balanced3' if model == 'Qwen3' else 'balanced4')
                            assert placement['gpu_expert_layers_by_device'] == {
                                f'CUDA{rank}': balanced for rank in range(4)}
                    case.update(status='PASS', selected_attempt=path.name, source_commit=status['source_commit'],
                                TTFT=summarize([row['TTFT'] for row in samples]),
                                TPOT=summarize([row['TPOT'] for row in samples]),
                                E2E=summarize([row['E2E'] for row in samples]),
                                memory=memory_receipt(path, cell, system, samples))
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
             'Each completed row has three unfiltered clean measurements. Times are seconds; TPOT is seconds per generated token and includes attention. See [MEMORY_AUDIT.md](MEMORY_AUDIT.md) for the expert budget and total HBM measurements.', '',
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
    lines.extend(['', 'DeepSeek MoE-Infinity uses EAM eviction priorities with speculative prefetch disabled after a native expert-wait stall; Qwen MoE-Infinity retains speculative EAM prefetch. See `DEEPSEEK_INFINITY_ADAPTATION.md`.'])
    (REPO / 'PROGRESS.md').write_text('\n'.join(lines) + '\n')
    memory_lines = ['# Main-table memory audit', '',
                    'C30 limits resident expert weights, not total HBM. MoE-Infinity expert peak charges are checked against that budget for every completed repeat. The HBM column is the highest 1 Hz supervisor sample across physical GPUs 0/1/4/5 during each job; it may miss shorter peaks. PyTorch allocated/reserved peaks come from worker counters where available. The difference between those counters does not isolate KV, attention, allocator, and native workspace costs. Pinned host memory is shown only when the worker measured it.', '',
                    '| Dataset | Model | B/rank | Input | System | C30 expert budget/GPU (GiB) | Sampled peak HBM/GPU (GiB) | PyTorch allocated peak/GPU (GiB) | PyTorch reserved peak/GPU (GiB) | Pinned host/rank (GiB) |',
                    '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for case in cases:
        if case['status'] != 'PASS':
            continue
        receipt = case['memory']
        def span(values, scale):
            if values is None:
                return 'not measured'
            converted = [value / scale for value in values]
            return f'{min(converted):.2f}–{max(converted):.2f}'
        memory_lines.append(
            f'| {case["dataset"]} | {case["model"]} | {case["local_batch"]} | '
            f'{case["input_tokens"]} | {DISPLAY[case["system"]]} | '
            f'{span(receipt["expert_budget_bytes_per_gpu"], 2**30)} | '
            f'{span(receipt["sampled_peak_hbm_mib_by_physical_gpu"].values(), 1024)} | '
            f'{span(receipt["torch_peak_allocated_bytes_by_rank"], 2**30)} | '
            f'{span(receipt["torch_peak_reserved_bytes_by_rank"], 2**30)} | '
            f'{span(receipt["pinned_host_bytes_by_rank"], 2**30)} |')
    (REPO / 'MEMORY_AUDIT.md').write_text('\n'.join(memory_lines) + '\n')
    print(f'{completed}/64 validated rows')


if __name__ == '__main__':
    main()
