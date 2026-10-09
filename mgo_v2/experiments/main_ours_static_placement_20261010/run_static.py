"""Run Static on every prior main_OURS policy-gap cell with trace-hash checks."""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GAP = ROOT / 'mgo_v2/experiments/policy_gap_c60_20261008'
OUTPUT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
RAW = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
GUARD = ROOT / 'mgo_v2/scripts/run_headline_job.py'


def cases():
    followup = json.loads((GAP / 'FOLLOWUP_RESULTS.json').read_text())['cases']
    original = json.loads((GAP / 'SHAREGPT_RESULTS.json').read_text())['cases']
    for old in original:
        yield old['cell'], 'nvswitch', old['label'], RAW / 'WORKLOADS.json'
    for old in followup:
        yield old['cell'], old['transport'], old['label'], RAW / 'FOLLOWUP_WORKLOADS.json'


def label(cell, transport):
    fields = cell.lower().split('_')
    return 'main_ours_static_' + '_'.join((fields[2], fields[3], fields[6], fields[7], transport)) + '_v1'


def check(old_label, new_label):
    original = OUTPUT / old_label
    measured = OUTPUT / new_label
    for rank in range(4):
        old = json.loads((original / f'trace_rank{rank}.json').read_text())
        new = json.loads((measured / f'trace_rank{rank}.json').read_text())
        assert old['route_sha256'] == new['route_sha256'], (old_label, new_label, rank)
        assert old['teacher_tokens'] == new['teacher_tokens'], (old_label, new_label, rank)
    result = json.loads((measured / 'result.json').read_text())
    prior = json.loads((original / 'result.json').read_text())
    assert result['status'] == 'PASS' and result['route_frozen']
    assert result['grouped_decode_mode'] == 'off' and result['prefetch'] == 'off'
    assert result['capture_policy'] == 'LA_CA_NEAR' and result['policy_modes'] == ['STATIC_MOD']
    assert result['main_capacities'] == prior['main_capacities']
    previous_status = json.loads((original / 'status.json').read_text())
    assert bool(result['nccl_p2p_disable']) == bool(previous_status['nccl_p2p_disable'])
    assert len(result['results']) == 2
    assert all(x['status'] == 'PASS' and x['backend'] == 'STATIC_MOD' for x in result['results'])


def main():
    cells = list(cases())
    env = os.environ.copy()
    env.update(MGO_NATIVE_POLICY='LA_CA_NEAR', MGO_NATIVE_COMPARE_POLICY='STATIC_ONLY',
               MGO_NATIVE_GROUPED_MODE='off', MGO_NATIVE_PREFETCH='off',
               MGO_NATIVE_MAIN_CAPACITY='1', MGO_NATIVE_DECODE_STEPS='32',
               MGO_MIN_GPU_FREE_MIB='2048', CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    env['PATH'] = '/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH']
    for index, (cell, transport, old_label, manifest) in enumerate(cells, 1):
        run_label = label(cell, transport)
        result = OUTPUT / run_label / 'result.json'
        if result.exists():
            check(old_label, run_label)
            print(f'[{index}/{len(cells)}] PASS already: {cell} {transport}', flush=True)
            continue
        print(f'[{index}/{len(cells)}] starting: {cell} {transport}', flush=True)
        command = [sys.executable, '-u', str(GUARD), '--workloads', str(manifest),
                   '--label', run_label, '--system', 'Ours',
                   '--worker', 'native_fullpath_model_worker.py', '--cell', cell,
                   '--timeout', '2400']
        if transport == 'p2p_disabled':
            command.append('--nccl-p2p-disable')
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if completed.returncode or not result.exists():
            raise SystemExit(f'cell failed: {cell} {transport}; inspect {OUTPUT / run_label}')
        check(old_label, run_label)
        print(f'[{index}/{len(cells)}] PASS: {cell} {transport}', flush=True)


if __name__ == '__main__':
    main()
