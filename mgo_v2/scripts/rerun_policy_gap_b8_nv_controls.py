"""Remeasure the two C30/B8 NVSwitch controls with P2P variable unset."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008')
RESULTS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
SUPERVISOR = Path(__file__).resolve().with_name('run_headline_job.py')


def main():
    manifest = ROOT / 'FOLLOWUP_WORKLOADS.json'
    cells = json.loads(manifest.read_text())['cells']
    selected = [cell for cell in cells if cell['capacity_percent'] == 30 and cell['local_batch'] == 8]
    assert len(selected) == 2
    env = dict(os.environ, MGO_NATIVE_POLICY='LA_CA_NEAR', MGO_NATIVE_COMPARE_POLICY='TRIPLE',
               MGO_NATIVE_PREFETCH='off', MGO_NATIVE_MAIN_CAPACITY='1', MGO_NATIVE_DECODE_STEPS='32',
               MGO_NATIVE_POLICY_DIAGNOSTIC='32', CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1',
               PATH='/home/hwlee/mgo-tools/native-expert-build/bin:' + os.environ['PATH'])
    for cell in selected:
        label = f"policy_gap_sharegpt_c30_nvswitch_b8_case{cell['case_id']}_v2_20261008"
        output = RESULTS / label
        if output.exists():
            status = json.loads((output / 'status.json').read_text())
            if status['status'] == 'PASS':
                assert all(json.loads((output / f'source_rank{rank}.json').read_text())['nccl_p2p_disable'] is None for rank in range(4))
                print('REUSE PASS', label, flush=True)
                continue
            raise RuntimeError(f'{label}: inspect existing {status["status"]} before retry')
        command = [PYTHON, str(SUPERVISOR), '--workloads', str(manifest), '--label', label,
                   '--system', 'Ours', '--worker', 'native_fullpath_model_worker.py',
                   '--cell', cell['cell'], '--timeout', '2400']
        print('START', label, flush=True)
        subprocess.run(command, env=env, check=True)
        status = json.loads((output / 'status.json').read_text())
        assert status['status'] == 'PASS'
        assert all(json.loads((output / f'source_rank{rank}.json').read_text())['nccl_p2p_disable'] is None for rank in range(4))
        print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
