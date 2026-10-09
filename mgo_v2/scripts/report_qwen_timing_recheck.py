"""Validate adjacent post-cleanup Qwen C20/C50 timing jobs."""

import json
import statistics
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/qwen_cache_ablation_20261009'
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
LABELS = {20: 'qca_recheck_c20_b16_r2_v1',
          30: 'qca_recheck_c30_b16_r2_v1',
          40: 'qca_recheck_c40_b16_r2_v1',
          50: 'qca_recheck_c50_b16_r2_v1'}


def read(path):
    return json.loads(path.read_text())


def main():
    progress = read(REPORT / 'PROGRESS.json')
    assert progress['status'] == 'PASS' and progress['completed'] == 4
    cases = []
    for percent, label in LABELS.items():
        primary = next(c for c in progress['cases'] if c['cache_percent'] == percent)
        original, new = JOBS / primary['selected_attempt'], JOBS / label
        old_status, status = read(original / 'status.json'), read(new / 'status.json')
        assert status['status'] == old_status['status'] == 'PASS'
        assert status['workload_manifest_sha256'] == old_status['workload_manifest_sha256']
        assert status['command'][status['command'].index('--repeats') + 1] == '2'
        assert '--smoke' not in status['command']
        for flag in ('--expert-executor', '--native-prefill', '--prefetch-off',
                     '--prefill-optimized', '--prefill-layout-fast',
                     '--decode-layout-fast', '--policy'):
            assert flag in status['command'] and flag in old_status['command']
        assert status['command'][status['command'].index('--policy') + 1] == 'LA_CA_NEAR'
        assert status['command'][status['command'].index('--expert-executor') + 1] == 'native'
        metrics = {name: [] for name in ('TTFT', 'TPOT', 'E2E')}
        h2d = []
        for repeat in (1, 2):
            batch = read(new / f'repeat{repeat}.json')
            assert batch['status'] == 'PASS' and batch['output_tokens'] == 64
            for name in metrics:
                metrics[name].append(batch[name])
            by_rank = 0
            for rank in range(4):
                measured = read(new / f'repeat{repeat}_rank{rank}.json')
                reference = read(original / f'repeat2_rank{rank}.json')
                assert measured['rank'] == rank
                assert measured['physical_gpu'] == (0, 1, 4, 5)[rank]
                assert measured['expert_cache_start'] == 'empty'
                assert measured['no_compile'] and measured['validation']['status'] == 'PASS'
                assert measured['tokens'] == reference['tokens']
                assert measured['request_ids'] == reference['request_ids']
                assert measured['validation']['state_hash'] == reference['validation']['state_hash']
                assert measured['validation']['role_hash'] == reference['validation']['role_hash']
                assert (measured['validation']['scheduler']['bytes'] ==
                        reference['validation']['scheduler']['bytes'])
                by_rank += measured['validation']['scheduler']['bytes']
            h2d.append(by_rank / 2**30)
        assert h2d[0] == h2d[1]
        cases.append(dict(cache_percent=percent, label=label,
                          source_commit=status['source_commit'],
                          metrics={name: dict(samples=values,
                                              mean=statistics.mean(values),
                                              minimum=min(values), maximum=max(values))
                                   for name, values in metrics.items()},
                          h2d_gib_all_ranks=h2d[0],
                          request_output_h2d_and_cache_match_original=True))
    c20, c50 = cases[0], cases[-1]
    gain = 1 - c50['metrics']['TPOT']['mean'] / c20['metrics']['TPOT']['mean']
    output = dict(status='PASS', model='Qwen3-30B-A3B-Instruct-2507',
                  dataset='ShareGPT', local_batch=16, input_tokens=512,
                  output_tokens=64, physical_gpus=[0, 1, 4, 5],
                  protocol='Sequential post-cleanup C20, C50, C30, C40 jobs; two unfiltered clean target repeats each',
                  mean_tpot_gain_c50_over_c20=gain, cases=cases)
    (REPORT / 'TIMING_RECHECK.json').write_text(json.dumps(output, indent=2) + '\n')
    for case in cases:
        print(f'C{case["cache_percent"]} TPOT {case["metrics"]["TPOT"]["samples"]}')
    print(f'PASS recheck C50 over C20 gain {100*gain:.2f}%')


if __name__ == '__main__':
    main()
