#!/usr/bin/env python3
"""Bounded six-cell R4 diagnostic launcher. Never launches R8."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-results/admission_trajectory_controller_breakdown_20261002')
INPUTS = Path('/home/hwlee/mgo-results/runtime_validation_20261001')
OUTPUT = PACKAGE / 'experiments/admission_trajectory_controller_breakdown_20261002'

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    cells = [dict(name=f'b{batch}_{policy}', batch=batch,
        config=dict(global_cache_ratio=.3, substitution_enabled=True, eviction='coverage',
                    same_layer_alpha=.25, path_eta=.5, seed=42, admission=policy))
        for batch in (4, 8, 16) for policy in ('random', 'hungarian_current')]
    cell_file = ROOT / 'cells.json'
    cell_file.write_text(json.dumps(cells, indent=2) + '\n')
    state_file = ROOT / 'status.json'
    assert not state_file.exists(), 'Use a new root or explicitly audit a failed launch before retrying'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,4,5', PYTHONPATH=str(PACKAGE),
        OMP_NUM_THREADS='4', MOE_EP_DISABLE_ARCHER_EVICT='1', MOE_EP_NATIVE_NUMERICS='1',
        MOE_EP_SLOT_VIEWS='1', MGO_STACK_DUMP_SECONDS='0')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
        str(PACKAGE / 'examples/diagnose_trajectory.py'), '--model', '/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507',
        '--offload-dir', str(INPUTS / 'expert_store'), '--similarity', str(INPUTS / 'similarity.npy'),
        '--affinity', str(INPUTS / 'affinity.npz'), '--workload', str(INPUTS / 'screen_workload.json'),
        '--cells', str(cell_file), '--output', str(ROOT / 'physical'), '--steps', '65', '--repeats', '1', '--debug-cache']
    source_paths = list((PACKAGE / 'mgo_v2').glob('*.py')) + [PACKAGE / 'examples/diagnose_trajectory.py', Path(__file__)]
    state = dict(status='RUNNING', plan_commit='30614bbc065c140060ef31af9bc4f8615ab16ca5',
        base_result_commit='e61758e2b25922d56e640735f27fef382b9dbac9',
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PACKAGE,text=True).strip(),
        started_unix=time.time(), world=4, physical_gpus=[0,1,4,5], cells=cells, command=command,
        source_sha256={str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [INPUTS/'similarity.npy',INPUTS/'affinity.npz',INPUTS/'screen_workload.json']},
        diagnostic_only=True, optional_profiles='omitted: reuse existing resident-only evidence; no new GPU timing attribution')
    state_file.write_text(json.dumps(state, indent=2)+'\n')
    (OUTPUT / 'measurement_manifest.json').write_text(json.dumps(state, indent=2)+'\n')
    with (ROOT / 'physical.log').open('w') as log:
        code = subprocess.call(command, cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(status='PHYSICAL_COMPLETE' if code == 0 else 'FAIL', exit_code=code, physical_finished_unix=time.time())
    state_file.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(dict(status=state['status'],exit_code=code)))
    raise SystemExit(code)

if __name__ == '__main__':
    main()
