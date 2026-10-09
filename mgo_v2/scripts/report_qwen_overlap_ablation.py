"""Validate paired normal versus H2D-serialized Qwen native execution."""

import json
import statistics
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
REPORT = PKG / 'experiments/qwen_cache_ablation_20261009'
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
LABELS = {'normal': 'qca_overlap_c20_normal_r2_v1',
          'serial': 'qca_overlap_c20_serial_r2_v1'}


def read(path):
    return json.loads(path.read_text())


def main():
    jobs = {name: JOBS / label for name, label in LABELS.items()}
    statuses = {name: read(path / 'status.json') for name, path in jobs.items()}
    assert all(status['status'] == 'PASS' for status in statuses.values())
    assert len({status['source_commit'] for status in statuses.values()}) == 1
    assert len({status['workload_manifest_sha256'] for status in statuses.values()}) == 1
    rows = {}
    for name, path in jobs.items():
        command = statuses[name]['command']
        assert command[command.index('--repeats') + 1] == '2' and '--smoke' not in command
        for flag in ('--native-prefill', '--prefetch-off', '--decode-layout-fast',
                     '--prefill-layout-fast', '--prefill-optimized'):
            assert flag in command
        assert ('--h2d-serial-ablation' in command) == (name == 'serial')
        assert command[command.index('--policy') + 1] == 'LA_CA_NEAR'
        assert command[command.index('--expert-executor') + 1] == 'native'
        metrics = {key: [] for key in ('TTFT', 'TPOT', 'E2E')}
        repeats = []
        for repeat in (1, 2):
            batch = read(path / f'repeat{repeat}.json')
            assert batch['status'] == 'PASS' and batch['output_tokens'] == 64
            for key in metrics:
                metrics[key].append(batch[key])
            ranks = []
            for rank in range(4):
                item = read(path / f'repeat{repeat}_rank{rank}.json')
                assert item['rank'] == rank and item['physical_gpu'] == (0, 1, 4, 5)[rank]
                assert item['expert_cache_start'] == 'empty' and item['no_compile']
                assert item['validation']['status'] == 'PASS'
                assert item['h2d_serial_ablation'] == (name == 'serial')
                counters = item['native_executor_counts']
                assert counters and counters['groups'] > 0 and counters['waves'] >= 3072
                if name == 'serial':
                    assert counters['serial_waited_copies'] > 0
                    assert counters['serial_wait_wall_ns'] > 0
                    assert counters['waits'] == 0
                else:
                    assert counters['serial_waited_copies'] == 0
                    assert counters['serial_wait_wall_ns'] == 0
                ranks.append(dict(rank=rank, physical_gpu=item['physical_gpu'],
                                  request_ids=item['request_ids'], tokens=item['tokens'],
                                  h2d_bytes=item['validation']['scheduler']['bytes'],
                                  state_hash=item['validation']['state_hash'],
                                  role_hash=item['validation']['role_hash'],
                                  native_executor_counts=counters))
            repeats.append(ranks)
        # The private token rows are held in memory only and excluded from Git.
        rows[name] = dict(label=path.name,
                          metrics={key: dict(samples=values,
                                             mean=statistics.mean(values),
                                             minimum=min(values), maximum=max(values))
                                   for key, values in metrics.items()},
                          repeats=repeats)
    for repeat in range(2):
        for rank in range(4):
            normal, serial = (rows[name]['repeats'][repeat][rank]
                              for name in ('normal', 'serial'))
            for key in ('request_ids', 'tokens', 'h2d_bytes', 'state_hash', 'role_hash'):
                assert normal[key] == serial[key], (repeat, rank, key)
            assert normal['native_executor_counts']['groups'] == serial['native_executor_counts']['groups']
    public = {}
    for name, row in rows.items():
        public[name] = dict(label=row['label'], metrics=row['metrics'],
                            ranks_per_repeat=[
                                [dict(rank=r['rank'], physical_gpu=r['physical_gpu'],
                                      h2d_gib=r['h2d_bytes'] / 2**30,
                                      native_executor_counts=r['native_executor_counts'])
                                 for r in repeat] for repeat in row['repeats']])
    normal = public['normal']['metrics']['TPOT']['mean']
    serial = public['serial']['metrics']['TPOT']['mean']
    output = dict(status='PASS', model='Qwen3-30B-A3B-Instruct-2507',
                  dataset='ShareGPT', physical_gpus=[0, 1, 4, 5], local_batch=16,
                  input_tokens=512, output_tokens=64, cache_percent=20,
                  source_commit=statuses['normal']['source_commit'],
                  exact_request_token_h2d_and_cache_parity=True,
                  normal_tpot_mean_s_per_token=normal,
                  serialized_tpot_mean_s_per_token=serial,
                  normal_advantage_ms_per_token=(serial - normal) * 1000,
                  serial_slowdown_percent=100 * (serial / normal - 1),
                  scope='Current-layer demand copies waited after dispatch before native expert compute; no added global barrier; dispatch still overlaps H2D',
                  arms=public)
    (REPORT / 'OVERLAP_ABLATION.json').write_text(json.dumps(output, indent=2) + '\n')
    print(f'PASS exact parity; normal {normal:.6f}, serial {serial:.6f} '
          f's/token; net advantage {(serial-normal)*1000:.2f} ms/token')


if __name__ == '__main__':
    main()
