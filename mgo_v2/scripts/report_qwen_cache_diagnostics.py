"""Validate C20/C50 Qwen cache/fetch diagnostics against clean main_OURS runs."""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/qwen_cache_ablation_20261009'
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
EXPERT_BYTES = 9 * 2**20
DECODE_LAYERS = 48 * 63
PHASES = ('moe.metadata', 'moe.current_controller', 'layout_cpu',
          'layout_device_materialization', 'moe.forward_a2a',
          'moe.forward_complete', 'moe.demand_h2d', 'required_h2d_exposed_wait',
          'moe.expert_compute', 'moe.return_a2a', 'return_token_a2a')


def read(path):
    return json.loads(path.read_text())


def percentile(values, fraction):
    data = sorted(values)
    return data[math.ceil(fraction * len(data)) - 1]


def find_diagnostic(percent):
    paths = sorted(JOBS.glob(f'qca_fetch_c{percent}_b16_r1_v*'),
                   key=lambda p: int(p.name.rsplit('_v', 1)[1]))
    passed = [p for p in paths if (p / 'status.json').exists()
              and read(p / 'status.json').get('status') == 'PASS'
              and all((p / f'generation_diagnostic_rank{rank}.json').exists()
                      for rank in range(4))]
    assert passed, f'C{percent} diagnostic incomplete'
    return passed[-1]


def main():
    progress = read(REPORT / 'PROGRESS.json')
    assert progress['status'] == 'PASS' and progress['completed'] == 4
    cases = []
    for percent in (20, 50):
        primary = next(c for c in progress['cases'] if c['cache_percent'] == percent)
        assert primary['status'] == 'PASS'
        ppath = JOBS / primary['selected_attempt']
        dpath = find_diagnostic(percent)
        status, reference = read(dpath / 'status.json'), read(ppath / 'status.json')
        assert status['workload_manifest_sha256'] == reference['workload_manifest_sha256']
        assert status['command'][status['command'].index('--repeats') + 1] == '1'
        assert '--post-generation-diagnostic' in status['command']
        for flag in ('--native-prefill', '--prefetch-off', '--decode-layout-fast'):
            assert flag in status['command'] and flag in reference['command']
        batch = read(dpath / 'repeat1.json')
        assert batch['status'] == 'PASS' and batch['output_tokens'] == 64
        ranks = []
        for rank in range(4):
            measured = read(dpath / f'repeat1_rank{rank}.json')
            diagnostic = read(dpath / f'generation_diagnostic_rank{rank}.json')
            original = [read(ppath / f'repeat{repeat}_rank{rank}.json')
                        for repeat in (1, 2, 3)]
            assert measured['prefetch_off'] and measured['expert_executor'] == 'native'
            assert measured['physical_gpu'] == (0, 1, 4, 5)[rank]
            assert all(measured['tokens'] == item['tokens'] for item in original)
            measured_h2d = measured['validation']['scheduler']['bytes']
            assert all(measured_h2d == item['validation']['scheduler']['bytes']
                       for item in original)
            assert diagnostic['status'] == 'PASS' and diagnostic['token_and_cache_parity']
            assert diagnostic['output']['tokens'] == measured['tokens']
            assert diagnostic['validation']['scheduler']['bytes'] == measured_h2d
            events = diagnostic['decode_cache_events']
            assert len(events) == DECODE_LAYERS
            assert [e['event_index'] for e in events] == list(range(48, 48 * 64))
            assert all(e['prefetch_hits'] == 0 for e in events)
            counts = {key: sum(e[key] for e in events)
                      for key in ('expert_uses', 'token_expert_uses', 'main_hits',
                                  'demand_misses', 'main_hit_tokens', 'miss_tokens',
                                  'surviving_prefill_hits')}
            assert counts['expert_uses'] == counts['main_hits'] + counts['demand_misses']
            copies = diagnostic['h2d_copies']
            assert len(copies) == diagnostic['validation']['scheduler']['copies']
            assert len(copies) * EXPERT_BYTES == measured_h2d
            service = [item['service_seconds'] * 1000 for item in copies]
            assert service and all(value > 0 for value in service)
            segments = defaultdict(float)
            for row in diagnostic['segments']:
                if row['step'] >= 1:
                    segments[row['phase']] += row['stream_seconds']
            ranks.append(dict(rank=rank, physical_gpu=(0, 1, 4, 5)[rank],
                              **counts,
                              hit_rate=counts['main_hits'] / counts['expert_uses'],
                              decode_h2d_gib=counts['demand_misses'] * EXPERT_BYTES / 2**30,
                              batch_h2d_gib=measured_h2d / 2**30,
                              copy_service_ms_median=statistics.median(service),
                              copy_service_ms_p95=percentile(service, .95),
                              copy_service_ms_max=max(service),
                              decode_phase_ms_per_token={key: segments[key] * 1000 / 63
                                                         for key in PHASES},
                              output_and_h2d_match_all_primary_repeats=True,
                              diagnostic_replay_token_and_cache_parity=True))
        total_uses = sum(r['expert_uses'] for r in ranks)
        total_hits = sum(r['main_hits'] for r in ranks)
        total_misses = sum(r['demand_misses'] for r in ranks)
        cases.append(dict(cache_percent=percent, primary_attempt=ppath.name,
                          diagnostic_attempt=dpath.name,
                          primary_tpot_median_s=primary['metrics']['TPOT']['median'],
                          clean_diagnostic_job_tpot_s=batch['TPOT'],
                          distinct_expert_uses=total_uses,
                          main_hits=total_hits, demand_misses=total_misses,
                          main_hit_rate=total_hits / total_uses,
                          decode_h2d_gib_all_ranks=total_misses * EXPERT_BYTES / 2**30,
                          batch_h2d_gib_all_ranks=sum(r['batch_h2d_gib'] for r in ranks),
                          max_rank_explicit_h2d_wait_ms_per_token=max(
                              r['decode_phase_ms_per_token']['required_h2d_exposed_wait']
                              for r in ranks),
                          ranks=ranks))
    output = dict(status='PASS', model='Qwen3-30B-A3B-Instruct-2507',
                  dataset='ShareGPT', physical_gpus=[0, 1, 4, 5], local_batch=16,
                  input_tokens=512, output_tokens=64, prefetch='OFF',
                  measurement='Post-generation replay is diagnostic, not primary timing',
                  cases=cases)
    (REPORT / 'FETCH_DIAGNOSTIC.json').write_text(json.dumps(output, indent=2) + '\n')
    for case in cases:
        print(f"C{case['cache_percent']}: hit {100*case['main_hit_rate']:.2f}%, "
              f"decode H2D {case['decode_h2d_gib_all_ranks']:.1f} GiB, "
              f"TPOT {case['primary_tpot_median_s']:.6f}s/token")


if __name__ == '__main__':
    main()
