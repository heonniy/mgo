"""Sequential guarded C20/C50 fixed-route jobs, three alternating pairs."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / 'run_headline_job.py'
ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
DATA = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009')
FIXED = DATA / 'critical_path_d1/FIXED_CONTINUATION.json'
ROUTES = DATA / 'critical_path_d1/route_capture_v1'
QUEUE_RECEIPT = DATA / 'critical_path_d1/QUEUE_STATUS.json'


def validate(label, capacity, fixed_sha, route_shas):
    directory = ROOT / label
    state = json.loads((directory / 'status.json').read_text())
    assert state['status'] == 'PASS', (label, state.get('error'))
    assert {row['gpu'] for row in state['restored']} == {0, 1, 4, 5}
    result = json.loads((directory / 'result.json').read_text())
    assert result['status'] == 'PASS' and not result['headline_eligible']
    assert result['cell'] == f'DeepSeekV2Lite_ShareGPT_R4_C{capacity}_B16_L512_O64'
    rows = []
    for rank in range(4):
        row = json.loads((directory / f'repeat1_rank{rank}.json').read_text())
        assert row['route_mode'] == 'replay'
        assert row['route_sha256'] == route_shas[rank]
        assert row['fixed_continuation_manifest_sha256'] == fixed_sha
        assert row['fixed_continuation'] and row['finite_logits']
        assert row['cache_start'] == 'empty' and row['output_tokens'] == 64
        assert row['decode_expert_groups'] > 0 and row['decode_expert_waves'] > 0
        rows.append(row)
    metric = json.loads((directory / 'repeat1.json').read_text())
    return {
        'label': label,
        'status': 'PASS',
        'capacity_percent': capacity,
        'TPOT_s_per_token': metric['TPOT'],
        'decode_h2d_gib': sum(row['decode_h2d_bytes'] for row in rows) / 2**30,
        'decode_expert_groups': sum(row['decode_expert_groups'] for row in rows),
        'decode_expert_waves': sum(row['decode_expert_waves'] for row in rows),
        'decode_dispatch_gib': sum(row['decode_dispatch_bytes'] for row in rows) / 2**30,
        'decode_return_gib': sum(row['decode_return_bytes'] for row in rows) / 2**30,
    }


def main():
    fixed_sha = hashlib.sha256(FIXED.read_bytes()).hexdigest()
    route_shas = [hashlib.sha256((ROUTES / f'route_rank{rank}.npz').read_bytes()).hexdigest()
                  for rank in range(4)]
    records = []
    for pair in (1, 2, 3):
        for capacity in (20, 50):
            label = f'dca_d1_fixedroute_c{capacity}_pair{pair}_v1'
            directory = ROOT / label
            if not directory.exists():
                command = [sys.executable, str(SCRIPT), '--label', label,
                           '--system', 'DeepSeek-D1-fixedroute',
                           '--worker', 'headline_ours_deepseek_worker.py',
                           '--cell', f'DeepSeekV2Lite_ShareGPT_R4_C{capacity}_B16_L512_O64',
                           '--workloads', str(DATA / 'WORKLOADS.json'),
                           '--repeats', '1', '--fixed-continuation', str(FIXED),
                           '--route-mode', 'replay', '--route-dir', str(ROUTES),
                           '--timeout', '3600']
                environment = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
                subprocess.run(command, env=environment, check=True)
            records.append(validate(label, capacity, fixed_sha, route_shas))
            tmp = QUEUE_RECEIPT.with_suffix('.tmp')
            tmp.write_text(json.dumps({'status': 'RUNNING', 'records': records}, indent=2) + '\n')
            os.replace(tmp, QUEUE_RECEIPT)
            print(json.dumps(records[-1]), flush=True)
    tmp = QUEUE_RECEIPT.with_suffix('.tmp')
    tmp.write_text(json.dumps({'status': 'PASS', 'records': records}, indent=2) + '\n')
    os.replace(tmp, QUEUE_RECEIPT)


if __name__ == '__main__':
    main()
