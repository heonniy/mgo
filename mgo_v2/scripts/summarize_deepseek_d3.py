"""Validate and summarize bounded DeepSeek D3 layer diagnostics."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
DATA = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009/critical_path_d1')
EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/deepseek_cache_ablation_20261009'
FIELDS = (
    'metadata_host_ms', 'metadata_recv_to_host_wait_ms',
    'controller_host_ms', 'layout_host_ms', 'h2d_enqueue_host_ms',
    'dense_submit_host_ms', 'dispatch_cuda_ms', 'expert_cuda_ms',
    'return_combine_cuda_ms', 'dispatch_to_combine_cuda_ms',
    'explicit_h2d_wait_cuda_ms', 'executor_host_ms',
    'ready_query_host_ms', 'native_wave_host_ms',
)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main():
    fixed_sha = hashlib.sha256((DATA / 'FIXED_CONTINUATION.json').read_bytes()).hexdigest()
    route_shas = [hashlib.sha256((DATA / 'route_capture_v1' / f'route_rank{r}.npz').read_bytes()).hexdigest()
                  for r in range(4)]
    layer_rows = []
    wave_rows = []
    cases = {}
    for capacity in (20, 50):
        label = f'dca_d3_fixedroute_c{capacity}_sample8_v1'
        directory = ROOT / label
        state = json.loads((directory / 'status.json').read_text())
        assert state['status'] == 'PASS' and {x['gpu'] for x in state['restored']} == {0, 1, 4, 5}
        metric = json.loads((directory / 'repeat1.json').read_text())
        ranks = [json.loads((directory / f'repeat1_rank{r}.json').read_text()) for r in range(4)]
        clean = ROOT / f'dca_d1_fixedroute_c{capacity}_pair1_v1'
        for rank, row in enumerate(ranks):
            clean_row = json.loads((clean / f'repeat1_rank{rank}.json').read_text())
            assert row['route_sha256'] == route_shas[rank]
            assert row['fixed_continuation_manifest_sha256'] == fixed_sha
            assert row['tokens'] == clean_row['tokens']
            assert row['decode_h2d_bytes'] == clean_row['decode_h2d_bytes']
            assert row['diagnostic_decode_steps'] == 8
            assert len(row['diagnostic_layers']) == 8 * 26
            assert row['finite_logits']
        layers = [row['diagnostic_layers'] for row in ranks]
        for index in range(8 * 26):
            group = [layers[r][index] for r in range(4)]
            step, layer = group[0]['step'], group[0]['layer']
            assert (step, layer) == (index // 26 + 1, index % 26)
            assert all((row['step'], row['layer'], row['rank']) == (step, layer, rank)
                       for rank, row in enumerate(group))
            item = {'cache_percent': capacity, 'step': step, 'layer': layer}
            for field in FIELDS:
                item['max_rank_' + field] = max(row[field] for row in group)
            item['max_ep_interval_rank'] = max(range(4),
                    key=lambda r: group[r]['dispatch_to_combine_cuda_ms'])
            item['last_return_host_submit_rank'] = max(range(4),
                    key=lambda r: group[r]['return_submit_ns'])
            for field in ('dispatch_submit_ns', 'return_submit_ns'):
                item[field + '_skew_ms'] = (max(row[field] for row in group) -
                                             min(row[field] for row in group)) / 1e6
            item['max_min_expert_token_row_ratio'] = (
                max(row['expert_token_rows'] for row in group) /
                min(row['expert_token_rows'] for row in group))
            item['local_fetches_all_ranks'] = sum(row['local_fetches'] for row in group)
            item['native_waves_all_ranks'] = sum(row['native_waves'] for row in group)
            item['expert_groups_all_ranks'] = sum(row['expert_groups'] for row in group)
            layer_rows.append(item)
        for step in range(1, 9):
            for rank in range(4):
                group = [row for row in layers[rank] if row['step'] == step]
                assert len(group) == 26
                wave_rows.append({
                    'cache_percent': capacity, 'step': step, 'rank': rank,
                    'physical_gpu': (0, 1, 4, 5)[rank],
                    'expert_groups': sum(row['expert_groups'] for row in group),
                    'expert_token_rows': sum(row['expert_token_rows'] for row in group),
                    'native_waves': sum(row['native_waves'] for row in group),
                    'local_fetches': sum(row['local_fetches'] for row in group),
                    'ready_query_host_ms': sum(row['ready_query_host_ms'] for row in group),
                    'native_wave_host_ms': sum(row['native_wave_host_ms'] for row in group),
                })
        subset = [row for row in layer_rows if row['cache_percent'] == capacity]
        clean_summary = json.loads((EXPERIMENT / 'MATCHED_ROUTE_VALIDATION.json').read_text())
        clean_median = clean_summary['cases'][str(capacity)]['tpot_median_s_per_token']
        cases[str(capacity)] = {
            'label': label,
            'source_commit': state['source_commit'],
            'diagnostic_tpot_s_per_token': metric['TPOT'],
            'diagnostic_overhead_vs_clean_median_percent':
                100 * (metric['TPOT'] / clean_median - 1),
            'sampled_step_layer_pairs': len(subset),
            'summary': {
                field: {
                    'median': statistics.median(row[field] for row in subset),
                    'p95': percentile([row[field] for row in subset], .95),
                }
                for field in ['max_rank_' + key for key in FIELDS] + [
                    'dispatch_submit_ns_skew_ms', 'return_submit_ns_skew_ms',
                    'max_min_expert_token_row_ratio']
            },
            'max_ep_interval_rank_frequency': dict(Counter(
                row['max_ep_interval_rank'] for row in subset)),
            'last_return_host_submit_rank_frequency': dict(Counter(
                row['last_return_host_submit_rank'] for row in subset)),
            'sampled_fetches': sum(row['local_fetches_all_ranks'] for row in subset),
            'sampled_waves': sum(row['native_waves_all_ranks'] for row in subset),
        }
    for filename, rows in (('LAYER_CRITICAL_PATH.csv', layer_rows),
                           ('EXECUTOR_WAVES.csv', wave_rows)):
        with (EXPERIMENT / filename).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    receipt = {
        'status': 'PASS',
        'scope': 'First eight decode steps, fixed continuation and frozen router; instrumented diagnostic only',
        'fixed_continuation_sha256': fixed_sha,
        'route_sha256_per_rank': route_shas,
        'cases': cases,
        'interpretation_limit': 'Per-phase max ranks may differ; CUDA intervals include host gaps/peer waits; receive-to-host wait can include NCCL completion; host submission skew is not GPU arrival skew.',
    }
    (EXPERIMENT / 'D3_VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': 'PASS', 'cases': {k: {
        'diagnostic_tpot_ms': v['diagnostic_tpot_s_per_token'] * 1000,
        'sampled_fetches': v['sampled_fetches'],
        'sampled_waves': v['sampled_waves'],
    } for k, v in cases.items()}}, indent=2))


if __name__ == '__main__':
    main()
