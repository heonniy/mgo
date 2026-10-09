"""Validate and summarize C20/C50 diagnostic-only DeepSeek cache runs."""

import json
import statistics
from pathlib import Path


ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
REPORT = Path(__file__).resolve().parents[1] / 'experiments/deepseek_cache_ablation_20261009'
PRIMARY = json.loads((REPORT / 'PROGRESS.json').read_text())
LABELS = {
    20: ('dca_plateau_c20_b16_r1_v1', 'dca_c20_b16_ours_r3_v1'),
    50: ('dca_plateau_c50_b16_r1_v1', 'dca_c50_b16_ours_r3_v1'),
}
SPAN_KEYS = ('route_to_dispatch_host_ms', 'dispatch_ms', 'expert_ms',
             'return_ms', 'h2d_wait_ms', 'h2d_host_wait_ms', 'rank_step_ms')


def read(path):
    return json.loads(path.read_text())


def main():
    assert PRIMARY['status'] == 'PASS' and PRIMARY['completed'] == 16
    cases = []
    for percent, (label, primary_label) in LABELS.items():
        path, reference = ROOT / label, ROOT / primary_label
        status = read(path / 'status.json')
        prior = read(reference / 'status.json')
        assert status['status'] == prior['status'] == 'PASS'
        assert status['workload_manifest_sha256'] == prior['workload_manifest_sha256']
        assert status['source_commit'] == '78477b089ce1d28019d20ec84ce75e41f7ce81b8'
        command = status['command']
        assert '--smoke' not in command and command[command.index('--repeats') + 1] == '1'
        summary = read(path / 'repeat1.json')
        primary = next(c for c in PRIMARY['cases'] if c['system'] == 'ours'
                       and c['cache_percent'] == percent)
        assert summary['status'] == 'PASS' and summary['output_tokens'] == 64
        ranks = []
        for rank in range(4):
            measured = read(path / f'repeat1_rank{rank}.json')
            original = read(reference / f'repeat2_rank{rank}.json')
            assert measured['tokens'] == original['tokens']
            assert measured['h2d_bytes'] == original['h2d_bytes']
            steps = measured['diagnostic_steps']
            assert len(steps) == 64 and [step['step'] for step in steps] == list(range(64))
            assert sum(step['h2d_bytes'] for step in steps) == measured['h2d_bytes']
            decode = steps[1:]
            row = dict(rank=rank, physical_gpu=(0, 1, 4, 5)[rank],
                       full_output_matches_primary=True, h2d_bytes_match_primary=True,
                       prefill_h2d_gib=steps[0]['h2d_bytes'] / 2**30,
                       decode_h2d_gib=sum(step['h2d_bytes'] for step in decode) / 2**30,
                       decode_expert_wait_calls=sum(step['expert_wait_calls'] for step in decode),
                       decode_expert_groups=sum(step['expert_groups'] for step in decode),
                       decode_expert_waves=sum(step['expert_waves'] for step in decode))
            for key in SPAN_KEYS:
                values = [step[key] for step in decode]
                row[key.replace('_ms', '_ms_per_token')] = statistics.mean(values)
                row[key.replace('_ms', '_ms_total')] = sum(values)
                row[key.replace('_ms', '_ms_step_max')] = max(values)
            ranks.append(row)
        cases.append(dict(cache_percent=percent, diagnostic_attempt=label,
                          diagnostic_source_commit=status['source_commit'],
                          primary_attempt=primary_label,
                          target_tpot_s_per_token=summary['TPOT'],
                          primary_median_tpot_s_per_token=primary['TPOT']['median'],
                          timing_overhead_percent=100 *
                          (summary['TPOT'] / primary['TPOT']['median'] - 1),
                          decode_h2d_gib_all_ranks=sum(r['decode_h2d_gib'] for r in ranks),
                          max_rank_h2d_wait_ms_per_token=max(
                              r['h2d_wait_ms_per_token'] for r in ranks),
                          max_rank_expert_wait_calls=max(
                              r['decode_expert_wait_calls'] for r in ranks),
                          ranks=ranks))
    output = dict(status='PASS', comparison='C20 versus C50',
                  model='DeepSeek-V2-Lite-Chat', dataset='ShareGPT', local_batch=16,
                  input_tokens=512, output_tokens=64, decode_intervals=63,
                  physical_gpus=[0, 1, 4, 5],
                  definition='H2D wait is exposed main-CUDA-stream wait-event time, not total DMA time',
                  cases=cases)
    target = REPORT / 'CACHE_PLATEAU_DIAG.json'
    target.write_text(json.dumps(output, indent=2) + '\n')
    print('PASS', target)
    for case in cases:
        print(f"C{case['cache_percent']}: decode H2D {case['decode_h2d_gib_all_ranks']:.1f} GiB; "
              f"max-rank exposed H2D wait {case['max_rank_h2d_wait_ms_per_token']:.3f} ms/token; "
              f"diagnostic TPOT {case['target_tpot_s_per_token']:.6f} s/token")


if __name__ == '__main__':
    main()
