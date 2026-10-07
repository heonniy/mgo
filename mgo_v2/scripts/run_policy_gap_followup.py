"""Guarded C30 NVSwitch, then C30/C60 P2P-disabled policy comparisons."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
RESULTS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SCRIPT = Path(__file__).resolve().with_name('run_headline_job.py')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'


def main():
    manifest = ROOT / 'FOLLOWUP_WORKLOADS.json'
    data = json.loads(manifest.read_text())
    assert data['status'] == 'FROZEN' and data['unique_seed_cases'] == 4
    cells = data['cells']
    env = dict(os.environ, MGO_NATIVE_POLICY='LA_CA_NEAR', MGO_NATIVE_COMPARE_POLICY='TRIPLE',
               MGO_NATIVE_PREFETCH='off', MGO_NATIVE_MAIN_CAPACITY='1', MGO_NATIVE_DECODE_STEPS='32',
               MGO_NATIVE_POLICY_DIAGNOSTIC='32', CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1',
               PATH='/home/hwlee/mgo-tools/native-expert-build/bin:' + os.environ['PATH'])
    for transport, capacities in [('nvswitch', (30,)), ('p2p_disabled', (30, 60))]:
        for capacity in capacities:
            for cell in cells:
                if cell['capacity_percent'] != capacity:
                    continue
                label = f"policy_gap_sharegpt_c{capacity}_{transport}_b{cell['local_batch']}_case{cell['case_id']}_20261008"
                path = RESULTS / label
                if path.exists():
                    status = json.loads((path / 'status.json').read_text())
                    if status['status'] == 'PASS':
                        assert status['nccl_p2p_disable'] == (transport == 'p2p_disabled')
                        print('REUSE PASS', label, flush=True)
                        continue
                    raise RuntimeError(f'{label} exists but status is {status["status"]}; inspect before retry')
                command = [PYTHON, str(SCRIPT), '--workloads', str(manifest), '--label', label,
                           '--system', 'Ours', '--worker', 'native_fullpath_model_worker.py',
                           '--cell', cell['cell'], '--timeout', '2400']
                if transport == 'p2p_disabled':
                    command.append('--nccl-p2p-disable')
                print('START', label, flush=True)
                subprocess.run(command, env=env, check=True)
                status = json.loads((path / 'status.json').read_text())
                assert status['status'] == 'PASS' and status['nccl_p2p_disable'] == (transport == 'p2p_disabled')
                print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
