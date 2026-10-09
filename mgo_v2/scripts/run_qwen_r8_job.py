"""Guard one R8 Qwen ShareGPT headline job and restore eight owned model loads."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import run_full_pinned_r4 as owner


PKG = Path(__file__).resolve().parents[1]
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009')
WORKLOADS = ROOT / 'WORKLOADS.json'
LOAD = Path('/home/hwlee/mgo-results/model_inference_load_20261003')
PHYSICAL = tuple(range(8))
CELL = 'Qwen3_ShareGPT_R8_C30_B16_L512_O64'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE = '/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
INFINITY = '/home/hwlee/mgo-tools/headline-r4/infinity-env/bin/python'
WORKERS = {
    'ours': ('headline_ours_worker.py', PYTHON, 7200),
    'deepspeed': ('headline_deepspeed_worker.py', BASE, 7200),
    'infinity': ('headline_infinity_worker.py', INFINITY, 7200),
    'llama': ('headline_llama_sync_worker.py', BASE, 28800),
}


def write(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(row, indent=2) + '\n')
    temp.replace(path)


def apps():
    uuids = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader'], text=True)
    mapping = {u.strip(): int(g.strip()) for g, u in
               (line.split(',') for line in uuids.splitlines())}
    processes = subprocess.check_output(['nvidia-smi',
                                         '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader,nounits'], text=True)
    return [(mapping[u.strip()], int(pid.strip())) for u, pid in
            (line.split(',') for line in processes.splitlines())]


def gpu_state():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free,temperature.gpu,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    return {int(g): dict(free_mib=int(free), temperature_c=int(temp),
                         utilization=int(util))
            for g, free, temp, util in
            ([field.strip() for field in line.split(',')]
             for line in output.splitlines())}


def stop_owned_loads(quiet_2367=False):
    rows = json.loads((LOAD / 'processes.json').read_text())
    expected = {0, 1, 4, 5} if quiet_2367 else set(PHYSICAL)
    reserved = expected | set(PHYSICAL)
    owned = [row for row in rows if row['gpu'] in expected and owner.owned_idle(row['pid'])]
    assert len(owned) == len(expected) and {row['gpu'] for row in owned} == expected, owned
    outsiders = [(gpu, pid) for gpu, pid in apps()
                if gpu in reserved and pid not in {row['pid'] for row in owned}]
    assert not outsiders, f'foreign compute process on reserved GPU: {outsiders}'
    for row in owned:
        os.kill(row['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 30
    while any(owner.owned_idle(row['pid']) for row in owned) and time.monotonic() < deadline:
        time.sleep(.5)
    for row in owned:
        if owner.owned_idle(row['pid']):
            os.kill(row['pid'], signal.SIGKILL)
    deadline = time.monotonic() + 30
    while any(gpu in reserved for gpu, _ in apps()) and time.monotonic() < deadline:
        time.sleep(1)
    assert not [(gpu, pid) for gpu, pid in apps() if gpu in reserved]
    # Prune stale receipts for loads intentionally stopped before this job.
    write(LOAD / 'processes.json', [row for row in rows
                                   if row not in owned and owner.owned_idle(row['pid'])])
    return [row['gpu'] for row in owned]


def restore_owned_loads(gpus):
    previous = json.loads((LOAD / 'processes.json').read_text())
    restored = []
    for gpu in gpus:
        if any(g == gpu for g, _ in apps()):
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (LOAD / f'gpu{gpu}.log').open('a') as log:
            process = subprocess.Popen([PYTHON, '-u',
                                        str(PKG / 'examples/model_inference_load.py')],
                                       env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        restored.append(dict(gpu=gpu, pid=process.pid))
    write(LOAD / 'processes.json', previous + restored)
    return restored


def command(system, output, smoke, repeats, ours_mode=None):
    worker, python, _ = WORKERS[system]
    command = [python, '-u']
    if system in ('ours', 'deepspeed'):
        command += ['-m', 'torch.distributed.run', '--standalone',
                    f'--nproc_per_node={len(PHYSICAL)}']
    command += [str(PKG / 'examples' / worker), '--cell', CELL,
                '--output', str(output), '--repeats', str(repeats)]
    if smoke:
        command += ['--smoke']
    if system == 'ours':
        command += ['--expert-executor', 'native', '--native-prefill',
                    '--prefetch-off', '--prefill-optimized', '--prefill-layout-fast',
                    '--decode-layout-fast', '--policy', 'LA_CA_NEAR']
        if ours_mode in ('B', 'C'):
            command += ['--grouped-decode-mode',
                        'serial_all' if ours_mode == 'B' else 'two_wave']
    elif system == 'llama':
        command += ['--threads', '32', '--cuda-graphs', 'off',
                    '--graph-reuse', 'off', '--expert-placement',
                    'balanced7' if len(PHYSICAL) == 2 else 'balanced1']
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=tuple(WORKERS), required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--ours-mode', choices=('A', 'B', 'C'))
    parser.add_argument('--quiet-2367', action='store_true',
                        help='Keep 2/3/6/7 idle and pause managed loads on 0/1/4/5')
    args = parser.parse_args()
    assert args.ours_mode is None or args.system == 'ours'
    manifest = json.loads(WORKLOADS.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == list(PHYSICAL)
    cell = manifest['cells'][0]
    assert cell['cell'] == CELL and cell['global_requests'] == len(PHYSICAL) * 16
    assert cell['local_batch'] == 16 and cell['input_tokens'] == 512
    assert cell['output_tokens'] == 64 and cell['cache_percent'] == 30
    for phase in ('warmup', 'target'):
        assert hashlib.sha256(Path(cell[phase]['path']).read_bytes()).hexdigest() == cell[phase]['sha256']
    assert owner.host_available() >= 384 * 2**30
    label = args.system if args.ours_mode is None else f'ours_{args.ours_mode.lower()}'
    output = ROOT / 'jobs' / f'{label}_{"smoke" if args.smoke else "full"}_v{args.attempt}'
    assert not output.exists(), f'preserve earlier attempt: {output}'
    output.mkdir(parents=True)
    worker, _, timeout = WORKERS[args.system]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, PHYSICAL)),
               MGO_V2_PHYSICAL_GPUS=','.join(map(str, PHYSICAL)),
               MGO_HEADLINE_WORKLOADS=str(WORKLOADS), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TORCHINDUCTOR_COMPILE_THREADS='2', PYTHONFAULTHANDLER='1',
               PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{PKG}:{PKG / "scripts"}:{PKG / "examples"}')
    for key in list(env):
        if key.startswith('NCCL_'):
            del env[key]
    env['NCCL_CUMEM_ENABLE'] = '0'
    if args.system == 'ours':
        env['PATH'] = '/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH']
        env.update(CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    if args.system == 'llama':
        env.update(OMP_THREAD_LIMIT='32', GGML_CUDA_DISABLE_GRAPHS='1',
                   LLAMA_GRAPH_REUSE_DISABLE='1')
    state = dict(status='RUNNING', system=args.system, ours_mode=args.ours_mode,
                 smoke=args.smoke,
                 repeats=args.repeats, physical_gpus=PHYSICAL, started=time.time(),
                 quiet_2367=args.quiet_2367,
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                       cwd=PKG.parent, text=True).strip(),
                 command=command(args.system, output, args.smoke, args.repeats, args.ours_mode),
                 workload_sha256=hashlib.sha256(WORKLOADS.read_bytes()).hexdigest())
    write(output / 'status.json', state)
    stopped = []
    process = None
    try:
        stopped = stop_owned_loads(args.quiet_2367)
        state['stopped_owned_load_gpus'] = stopped
        assert min(x['free_mib'] for x in gpu_state().values()) >= 2048
        with (output / 'run.log').open('w') as log, (output / 'resources.jsonl').open('w') as resources:
            process = subprocess.Popen(state['command'], env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state['pid'] = process.pid
            write(output / 'status.json', state)
            while process.poll() is None:
                if time.time() - state['started'] > timeout:
                    raise TimeoutError(f'{args.system} exceeded {timeout} seconds')
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
                resources.write(json.dumps(dict(unix=time.time(), phase=phase,
                                                gpus=gpu, host_available=available)) + '\n')
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
        deadline = time.monotonic() + 30
        while any(g in PHYSICAL for g, _ in apps()) and time.monotonic() < deadline:
            time.sleep(1)
        state['remaining_gpu_processes'] = [(g, pid) for g, pid in apps() if g in PHYSICAL]
        state['restored_model_loads'] = restore_owned_loads(stopped)
        state['finished'] = time.time()
        write(output / 'status.json', state)


if __name__ == '__main__':
    main()
