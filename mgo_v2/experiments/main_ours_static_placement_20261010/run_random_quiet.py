"""One seeded, layer-balanced random owner baseline on the quiet R4 cell."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
REFERENCE = RESULTS / 'main_ours_c30_b8_s14_d5_env2_quiet_pair_v1'
LABEL = 'main_ours_c30_b8_s14_d5_env2_quiet_random_v1'
OUTPUT = RESULTS / LABEL
CELL = 'ShareGPT_R4_C30_B8_L128_O33_s14_d5'
MANIFEST = Path('/home/hwlee/mgo-results/policy_gap_c60_20261008/FOLLOWUP_WORKLOADS.json')


def validate():
    result = json.loads((OUTPUT / 'result.json').read_text())
    assert result['status'] == 'PASS' and result['route_frozen']
    assert result['policy_modes'] == ['RANDOM_HASH']
    assert result['capture_policy'] == 'LA_CA_NEAR'
    assert result['grouped_decode_mode'] == 'off' and result['prefetch'] == 'off'
    assert result['nccl_p2p_disable'] == '1'
    assert len(result['results']) == 2
    assert all(row['status'] == 'PASS' and row['backend'] == 'RANDOM_HASH'
               for row in result['results'])
    for rank in range(4):
        old = json.loads((REFERENCE / f'trace_rank{rank}.json').read_text())
        new = json.loads((OUTPUT / f'trace_rank{rank}.json').read_text())
        assert old['route_sha256'] == new['route_sha256']
        assert old['teacher_tokens'] == new['teacher_tokens']
    print(json.dumps(result['results'], indent=2), flush=True)


def main():
    assert not OUTPUT.exists(), f'preserve earlier attempt: {OUTPUT}'
    env = os.environ.copy()
    env.update(MGO_NATIVE_POLICY='LA_CA_NEAR',
               MGO_NATIVE_COMPARE_POLICY='RANDOM_ONLY',
               MGO_NATIVE_GROUPED_MODE='off',
               MGO_NATIVE_PREFETCH='off',
               MGO_NATIVE_MAIN_CAPACITY='1',
               MGO_NATIVE_DECODE_STEPS='32',
               MGO_MIN_GPU_FREE_MIB='2048',
               CUDA_HOME='/usr/local/cuda',
               TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    env['PATH'] = '/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH']
    command = [sys.executable, '-u', str(ROOT / 'mgo_v2/scripts/run_headline_job.py'),
               '--workloads', str(MANIFEST), '--label', LABEL,
               '--system', 'Ours', '--worker', 'native_fullpath_model_worker.py',
               '--cell', CELL, '--timeout', '2400', '--nccl-p2p-disable']
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    validate()


if __name__ == '__main__':
    main()
