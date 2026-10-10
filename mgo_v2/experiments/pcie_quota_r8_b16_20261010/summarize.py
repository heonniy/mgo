"""Audit and report the matched R8/B16 PCIe quota comparison."""

import hashlib
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
POLICIES = (
    ('near_quota', 'LA_CA_NEAR', 'Original Near'),
    ('fast_quota', 'NEAR_FAST', 'Fast-rank quota'),
    ('pcie_quota', 'NEAR_PCIE', 'PCIe lookup quota'),
)
METRICS = ('TTFT', 'TPOT', 'E2E', 'throughput')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_policy(label, policy, display, manifest_sha, table_sha):
    primary = RAW / 'jobs' / f'{label}_full_v1'
    status = json.loads((primary / 'status.json').read_text())
    result = json.loads((primary / 'result.json').read_text())
    assert status['status'] == result['status'] == 'PASS'
    assert status['repeats'] == 2 and not status['smoke'] and status['quiet_2367']
    assert status['physical_gpus'] == list(range(8))
    assert status['workload_sha256'] == manifest_sha
    assert result['policy'] == policy and result['expert_executor'] == 'native'
    command = status['command']
    assert command[command.index('--policy') + 1] == policy
    assert '--prefetch-off' in command and '--decode-layout-fast' in command
    if policy != 'LA_CA_NEAR':
        table = Path(command[command.index('--quota-table') + 1])
        assert sha(table) == table_sha
    rows = [json.loads((primary / f'repeat{i}.json').read_text()) for i in (1, 2)]
    gaps = {key: 100 * abs(rows[0][key] - rows[1][key]) /
            statistics.mean(row[key] for row in rows) for key in ('TPOT', 'E2E')}
    paths = [primary]
    if max(gaps.values()) > 2:
        extra = RAW / 'jobs' / f'{label}_full_v2'
        extra_status = json.loads((extra / 'status.json').read_text())
        assert extra_status['status'] == 'PASS' and extra_status['repeats'] == 1
        assert extra_status['workload_sha256'] == manifest_sha
        assert extra_status['physical_gpus'] == list(range(8))
        rows.append(json.loads((extra / 'repeat1.json').read_text()))
        paths.append(extra)
    else:
        assert not (RAW / 'jobs' / f'{label}_full_v2').exists()
    rank_receipts = []
    for path in paths:
        repeats = json.loads((path / 'status.json').read_text())['repeats']
        for i in range(1, repeats + 1):
            items = [json.loads((path / f'repeat{i}_rank{rank}.json').read_text())
                     for rank in range(8)]
            assert all(row['policy'] == policy and row['expert_executor'] == 'native'
                       and row['prefetch_off'] and row['no_compile']
                       and row['validation']['status'] == 'PASS'
                       and row['validation']['controller']['quota_violations'] == 0
                       and row['validation']['physical_slots'] == 1843
                       for row in items)
            assert len({row['validation']['controller']['mandatory'] for row in items}) == 1
            rank_receipts.append(dict(path=str(path), repeat=i,
                                      token_hashes=[row['argmax_hash'] for row in items],
                                      h2d_copies=[row['validation']['scheduler']['copies'] for row in items],
                                      h2d_bytes=[row['validation']['scheduler']['bytes'] for row in items],
                                      mandatory_misses=items[0]['validation']['controller']['mandatory']))
    assert len(rank_receipts) == len(rows)
    metrics = {}
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        metrics[key] = dict(samples=values,
                            reported=statistics.median(values) if len(values) == 3
                            else statistics.mean(values),
                            minimum=min(values), maximum=max(values))
    return dict(label=display, policy=policy, paths=list(map(str, paths)),
                source_commits=[json.loads((p / 'status.json').read_text())['source_commit']
                                for p in paths], first_pair_gap_pct=gaps,
                initial_pair_above_5pct=max(gaps.values()) > 5,
                metrics=metrics, rank_receipts=rank_receipts)


def fmt(row, metric):
    data = row['metrics'][metric]
    places = 4 if metric == 'TPOT' else 3
    return (f"{data['reported']:.{places}f} "
            f"[{data['minimum']:.{places}f}, {data['maximum']:.{places}f}]")


def main():
    state = json.loads((HERE / 'PHYSICAL_STATUS.json').read_text())
    assert state['status'] == 'PASS'
    table_path = HERE / 'LOOKUP.json'
    table = json.loads(table_path.read_text())
    assert table['status'] == 'PASS'
    manifest = RAW / 'WORKLOADS.json'
    rows = [summarize_policy(label, policy, name, sha(manifest), sha(table_path))
            for label, policy, name in POLICIES]
    reference = rows[0]['rank_receipts'][0]['token_hashes']
    for row in rows:
        row['token_parity_with_original_near'] = all(
            receipt['token_hashes'] == reference for receipt in row['rank_receipts'])
    output = dict(status='PASS', physical_gpus=list(range(8)),
                  workload_sha256=sha(manifest), lookup_sha256=sha(table_path),
                  policies=rows)
    (HERE / 'RESULTS.json').write_text(json.dumps(output, indent=2) + '\n')
    changed = [case for case in table['cases'] if case['pcie_quota'] != case['fast_quota']]
    lines = ['# R8/B16 PCIe quota experiment', '',
             'Qwen3-30B, ShareGPT, per-GPU B16, input512, output64, C30; '
             'native Ready-First `main_OURS`, compiled metadata/layout, '
             'prefetch OFF. All policies use the same frozen requests and '
             'balanced miss-count quota. The lookup changes only which ranks '
             'receive the extra misses. Every primary target starts with an '
             'empty expert cache.', '',
             '| Policy | Repeats | TTFT (s) | TPOT (s/token) | E2E (s) | TPS | Token parity |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['label']} | {len(row['metrics']['E2E']['samples'])} | "
                     f"{fmt(row,'TTFT')} | {fmt(row,'TPOT')} | {fmt(row,'E2E')} | "
                     f"{fmt(row,'throughput')} | "
                     f"{'exact' if row['token_parity_with_original_near'] else 'DIFF'} |")
    lines += ['', 'Values are the mean of two or median of three unfiltered '
              'target repeats; brackets contain the full range. A third was '
              'added only when the first-pair TPOT or E2E difference exceeded '
              '2%. The initial-pair gap and all raw receipt paths are in '
              '[RESULTS.json](RESULTS.json).', '',
              f"The PCIe and simple fast-rank lookup rows differ for {len(changed)} "
              'of 129 miss counts. At 12 misses both select '
              f"`{table['cases'][12]['pcie_quota']}`. This is a measured "
              'H2D-service model, not a guarantee of end-to-end gain. '
              'The microbenchmark uses pinned 9 MiB copies with no model, '
              'NCCL or expert computation. Raw calibration records are under '
              '`/home/hwlee/mgo-results/pcie_quota_r8_b16_20261010/`.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
