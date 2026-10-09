"""Run the bounded frozen-route grouped policy screen one guarded cell at a time."""

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MANIFEST = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008/FOLLOWUP_WORKLOADS.json')
OUTPUT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
GUARD = ROOT / 'mgo_v2/scripts/run_headline_job.py'
WORKER = 'native_fullpath_model_worker.py'


def label(cell):
    fields = cell.lower().split('_')
    return 'grouped_frozen_policy_' + '_'.join((fields[2], fields[3], fields[6], fields[7])) + '_v1'


def main():
    cells = [row['cell'] for row in json.loads(MANIFEST.read_text())['cells']]
    env = os.environ.copy()
    env.update(MGO_NATIVE_POLICY='BR', MGO_NATIVE_COMPARE_POLICY='BR_NEAR_STATIC',
               MGO_NATIVE_GROUPED_MODE='hit_then_miss', MGO_NATIVE_PREFETCH='off',
               MGO_NATIVE_MAIN_CAPACITY='1', MGO_NATIVE_DECODE_STEPS='32',
               MGO_MIN_GPU_FREE_MIB='2048', CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    env['PATH'] = '/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH']
    for index, cell in enumerate(cells, 1):
        run_label = label(cell)
        result = OUTPUT / run_label / 'result.json'
        if result.exists() and json.loads(result.read_text()).get('status') == 'PASS':
            print(f'[{index}/{len(cells)}] PASS already: {cell}', flush=True)
            continue
        print(f'[{index}/{len(cells)}] starting: {cell}', flush=True)
        command = [sys.executable, '-u', str(GUARD), '--workloads', str(MANIFEST),
                   '--label', run_label, '--system', 'Ours', '--worker', WORKER,
                   '--cell', cell, '--timeout', '2400']
        completed = subprocess.run(command, cwd=ROOT, env=env)
        if completed.returncode or not result.exists() or json.loads(result.read_text()).get('status') != 'PASS':
            raise SystemExit(f'cell failed: {cell}; inspect {OUTPUT / run_label}')
        print(f'[{index}/{len(cells)}] PASS: {cell}', flush=True)


if __name__ == '__main__':
    main()
