"""Run prepared policy cases sequentially under the guarded R4 supervisor."""
import argparse
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
RESULTS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SCRIPT = Path(__file__).resolve().with_name('run_headline_job.py')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('ShareGPT', 'LMSYS'), required=True)
    parser.add_argument('--skip', nargs='*', default=[])
    args = parser.parse_args()
    workload = ROOT / ('WORKLOADS.json' if args.dataset == 'ShareGPT' else 'LMSYS_WORKLOADS.json')
    cases = json.loads(workload.read_text())['cells']
    env = dict(os.environ, MGO_NATIVE_POLICY='LA_CA_NEAR', MGO_NATIVE_COMPARE_POLICY='TRIPLE',
               MGO_NATIVE_PREFETCH='off', MGO_NATIVE_MAIN_CAPACITY='1', MGO_NATIVE_DECODE_STEPS='32',
               MGO_NATIVE_POLICY_DIAGNOSTIC='32', CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1',
               PATH='/home/hwlee/mgo-tools/native-expert-build/bin:' + os.environ['PATH'])
    for cell in cases:
        if cell['cell'] in args.skip:
            continue
        label = f"policy_gap_{args.dataset.lower()}_b{cell['local_batch']}_case{cell['case_id']}_20261008"
        path = RESULTS / label
        if path.exists():
            status = json.loads((path / 'status.json').read_text())
            if status['status'] == 'PASS':
                print('REUSE PASS', label, flush=True)
                continue
            raise RuntimeError(f'{label} exists but status is {status["status"]}; inspect before retry')
        command = [PYTHON, str(SCRIPT), '--workloads', str(workload), '--label', label,
                   '--system', 'Ours', '--worker', 'native_fullpath_model_worker.py',
                   '--cell', cell['cell'], '--timeout', '2400']
        print('START', label, flush=True)
        subprocess.run(command, env=env, check=True)
        assert json.loads((path / 'status.json').read_text())['status'] == 'PASS'
        print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
