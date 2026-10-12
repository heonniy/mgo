"""Guard one R8 DeepSeek-V2-Lite ShareGPT main_OURS job on GPUs 0-7.

Reuses the Qwen R8 guard helpers. All eight GPUs are reserved: owned model loads
are stopped there, and any foreign compute process on any GPU aborts the job.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import run_full_pinned_r4 as owner
import contention_hogs
from run_qwen_r8_job import apps, gpu_state, restore_owned_loads, stop_owned_loads, write

PKG = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012')
WORKLOADS = ROOT / 'WORKLOADS.json'
PHYSICAL = tuple(range(8))
CELL = 'DeepSeekV2Lite_ShareGPT_R8_C30_B16_L512_O64'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
WORKER = 'headline_ours_deepseek_r8_worker.py'
POLICIES = ('BR', 'LA_CA_NEAR', 'NEAR_FAST', 'FAST_WORST', 'FAST_RANDOM', 'RANDOM_QUOTA_NEAR', 'BALANCED_RANDOM_NEAR', 'HAQ', 'HAQ_FAST', 'RANDOM', 'NEAR_SPLIT')
TABLE_POLICIES = ('NEAR_FAST', 'FAST_WORST', 'FAST_RANDOM', 'NEAR_SPLIT')
TIMEOUT = 7200


def command(output, args):
    cmd = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone',
           f'--nproc_per_node={len(PHYSICAL)}', str(PKG / 'examples' / WORKER),
           '--cell', CELL, '--output', str(output), '--repeats', str(args.repeats),
           '--decode-policy', args.decode_policy, '--prefill-policy', args.prefill_policy]
    if args.smoke:
        cmd += ['--smoke']
    if args.quota_table is not None:
        cmd += ['--quota-table', str(args.quota_table)]
    if args.teacher_tokens is not None:
        cmd += ['--teacher-tokens', str(args.teacher_tokens)]
    if args.inline_demand_h2d:
        cmd += ['--inline-demand-h2d']
    if args.h2d_serial_ablation:
        cmd += ['--h2d-serial-ablation']
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--decode-policy', choices=POLICIES, required=True)
    p.add_argument('--prefill-policy', choices=('BR', 'LA_CA_NEAR'), default='BR')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--attempt', type=int, default=1)
    p.add_argument('--quota-table', type=Path)
    p.add_argument('--teacher-tokens', type=Path)
    p.add_argument('--inline-demand-h2d', action='store_true')
    p.add_argument('--h2d-serial-ablation', action='store_true')
    p.add_argument('--job-label', required=True)
    contention_hogs.add_arguments(p)
    args = p.parse_args()
    assert (args.quota_table is not None) == (args.decode_policy in TABLE_POLICIES)
    assert args.teacher_tokens is None or not args.smoke
    for name in ('quota_table', 'teacher_tokens'):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
            assert getattr(args, name).is_file()
    manifest = json.loads(WORKLOADS.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == list(PHYSICAL)
    cell = manifest['cells'][0]
    assert cell['cell'] == CELL and cell['global_requests'] == len(PHYSICAL) * 16
    for phase in ('warmup', 'target'):
        assert hashlib.sha256(Path(cell[phase]['path']).read_bytes()).hexdigest() == cell[phase]['sha256']
    assert owner.host_available() >= 384 * 2**30
    label = args.job_label
    assert label.replace('_', '').isalnum() and label.islower()
    output = ROOT / 'jobs' / f'{label}_{"smoke" if args.smoke else "full"}_v{args.attempt}'
    assert not output.exists(), f'preserve earlier attempt: {output}'
    output.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, PHYSICAL)),
               MGO_V2_PHYSICAL_GPUS=','.join(map(str, PHYSICAL)),
               MGO_HEADLINE_WORKLOADS=str(WORKLOADS), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TORCHINDUCTOR_COMPILE_THREADS='2', PYTHONFAULTHANDLER='1',
               NUMBA_CACHE_DIR='/tmp/mgo-deepseek-r8-numba-v1',
               PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{PKG}:{PKG / "scripts"}:{PKG / "examples"}')
    for key in list(env):
        if key.startswith('NCCL_'):
            del env[key]
    env['NCCL_CUMEM_ENABLE'] = '0'
    env['PATH'] = '/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH']
    env.update(CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    state = dict(status='RUNNING', system='ours_deepseek_r8', smoke=args.smoke, repeats=args.repeats,
                 decode_policy=args.decode_policy, prefill_policy=args.prefill_policy,
                 inline_demand_h2d=args.inline_demand_h2d, h2d_serial_ablation=args.h2d_serial_ablation,
                 physical_gpus=PHYSICAL, started=time.time(),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PKG.parent, text=True).strip(),
                 source_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                                           cwd=PKG.parent, text=True).strip()),
                 command=command(output, args),
                 workload_sha256=hashlib.sha256(WORKLOADS.read_bytes()).hexdigest())
    write(output / 'status.json', state)
    stopped, process = [], None
    try:
        stopped = stop_owned_loads(False)
        state['stopped_owned_load_gpus'] = stopped
        assert min(x['free_mib'] for x in gpu_state().values()) >= 2048
        hogs = contention_hogs.start(args, PYTHON, env, output)
        state['contention_hogs'] = dict(gpus=hogs['gpus'], streams=hogs['streams'], gibps=hogs['gibps'])
        with (output / 'run.log').open('w') as log, (output / 'resources.jsonl').open('w') as resources:
            process = subprocess.Popen(state['command'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            state['pid'] = process.pid
            write(output / 'status.json', state)
            while process.poll() is None:
                if time.time() - state['started'] > TIMEOUT:
                    raise TimeoutError(f'exceeded {TIMEOUT} seconds')
                if (ROOT / 'STOP').exists() or (output / 'STOP').exists():
                    raise RuntimeError('owner STOP')
                available = owner.host_available()
                if available < 96 * 2**30:
                    raise RuntimeError('host available below 96 GiB')
                gpu = gpu_state()
                if min(x['free_mib'] for x in gpu.values()) < 2048:
                    raise RuntimeError('GPU free memory below 2 GiB')
                if max(x['temperature_c'] for x in gpu.values()) >= 85:
                    raise RuntimeError('GPU temperature reached 85 C')
                phase = json.loads((output / 'phase.json').read_text()) if (output / 'phase.json').exists() else {'phase': 'loading'}
                resources.write(json.dumps(dict(unix=time.time(), phase=phase, gpus=gpu,
                                                host_available=available)) + '\n')
                resources.flush()
                time.sleep(1)
        state['worker_returncode'] = process.returncode
        assert process.returncode == 0, f'worker exited {process.returncode}; see run.log'
        result = json.loads((output / 'result.json').read_text())
        assert result['status'] == 'PASS'
        state['status'] = 'PASS'
        state['result'] = result
    except BaseException as error:
        state.update(status='FAIL', error=repr(error))
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        if 'hogs' in locals():
            state['contention_hogs'] = contention_hogs.stop(hogs, output)
        deadline = time.monotonic() + 30
        while any(g in PHYSICAL for g, _ in apps()) and time.monotonic() < deadline:
            time.sleep(1)
        state['remaining_gpu_processes'] = [(g, pid) for g, pid in apps() if g in PHYSICAL]
        state['restored_model_loads'] = restore_owned_loads(stopped)
        state['finished'] = time.time()
        write(output / 'status.json', state)


if __name__ == '__main__':
    main()
