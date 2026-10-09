"""Guard four-rank Qwen runs while keeping all eight owned loads quiet."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import run_qwen_r8_job as guard


ROOT = Path('/home/hwlee/mgo-results/expert_grouping_ablation_20261009')
WORKLOADS = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
MAIN_TABLE_WORKLOADS = Path('/home/hwlee/mgo-results/main_table_2x2_20261008/ShareGPT/WORKLOADS.json')
B64_ROOT = Path('/home/hwlee/mgo-results/ep_overhead_r4_b64_20261009')
PHYSICAL = (0, 1, 4, 5)
CELL = 'Qwen3_ShareGPT_R4_C30_B16_L512_O64'
B64_CELL = 'Qwen3_ShareGPT_R4_C30_B64_L512_O64'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A', 'B', 'C'), required=True)
    parser.add_argument('--cell', choices=(CELL, B64_CELL), default=CELL)
    parser.add_argument('--runtime-root', type=Path, default=guard.PKG.parent)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--post-prefill-diagnostic', action='store_true')
    parser.add_argument('--compiled-dense', action='store_true')
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    runtime_pkg = args.runtime_root.resolve() / 'mgo_v2'
    assert (runtime_pkg / 'examples/headline_ours_worker.py').is_file()
    assert not (args.smoke and args.diagnostic)
    assert not args.post_prefill_diagnostic or args.diagnostic
    assert not args.diagnostic or args.repeats == 1
    workload_path = WORKLOADS if args.cell == CELL else MAIN_TABLE_WORKLOADS
    root = ROOT if args.cell == CELL else B64_ROOT
    manifest = json.loads(workload_path.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == list(PHYSICAL)
    spec, = [row for row in manifest['cells'] if row['cell'] == args.cell]
    assert (spec['local_batch'], spec['global_requests'], spec['input_tokens'],
            spec['output_tokens'], spec['cache_percent'], spec['expert_slots']) == (
                (16, 64, 512, 64, 30, 1843) if args.cell == CELL
                else (64, 256, 512, 64, 30, 1843))
    for phase in ('warmup', 'target'):
        source = spec[phase]
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    assert guard.owner.host_available() >= 384 * 2**30
    kind = 'smoke' if args.smoke else 'diagnostic' if args.diagnostic else 'full'
    output = root / 'jobs' / f'r4_{args.arm.lower()}_{kind}_v{args.attempt}'
    assert not output.exists(), f'preserve previous attempt: {output}'
    output.mkdir(parents=True)
    command = [guard.PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', str(runtime_pkg / 'examples/headline_ours_worker.py'),
               '--cell', args.cell, '--output', str(output), '--repeats', str(args.repeats),
               '--expert-executor', 'native', '--native-prefill', '--prefetch-off',
               '--prefill-optimized', '--prefill-layout-fast', '--decode-layout-fast',
               '--policy', 'LA_CA_NEAR']
    if args.arm in ('B', 'C'):
        command += ['--grouped-decode-mode', 'serial_all' if args.arm == 'B' else 'two_wave']
    if args.compiled_dense:
        command += ['--compiled-dense']
    if args.smoke:
        command += ['--smoke']
    if args.diagnostic:
        command += ['--post-generation-diagnostic']
    if args.post_prefill_diagnostic:
        command += ['--post-prefill-diagnostic']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, PHYSICAL)),
               MGO_V2_PHYSICAL_GPUS=','.join(map(str, PHYSICAL)),
               MGO_HEADLINE_WORKLOADS=str(workload_path), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               TORCHINDUCTOR_COMPILE_THREADS='2', PYTHONFAULTHANDLER='1',
               PYTHONPATH=f'/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003/cpu_deps:{runtime_pkg}:{runtime_pkg / "scripts"}:{runtime_pkg / "examples"}')
    for key in list(env):
        if key.startswith('NCCL_'):
            del env[key]
    env.update(NCCL_CUMEM_ENABLE='0',
               PATH='/home/hwlee/mgo-tools/native-expert-build/bin:' + env['PATH'],
               CUDA_HOME='/usr/local/cuda', TORCH_CUDA_ARCH_LIST='9.0', MAX_JOBS='1')
    state = dict(status='RUNNING', arm=args.arm, kind=kind,
                 physical_gpus=PHYSICAL, command=command, started=time.time(),
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                       cwd=args.runtime_root, text=True).strip(),
                 runtime_root=str(args.runtime_root.resolve()),
                 workload_sha256=hashlib.sha256(workload_path.read_bytes()).hexdigest())
    guard.write(output / 'status.json', state)
    stopped = []
    process = None
    try:
        stopped = guard.stop_owned_loads()
        assert min(guard.gpu_state()[gpu]['free_mib'] for gpu in PHYSICAL) >= 2048
        with (output / 'run.log').open('w') as log, (output / 'resources.jsonl').open('w') as resources:
            process = subprocess.Popen(command, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state['pid'] = process.pid
            guard.write(output / 'status.json', state)
            while process.poll() is None:
                if time.time() - state['started'] > 7200:
                    raise TimeoutError('R4 grouping job exceeded 7200 seconds')
                if (root / 'STOP').exists() or (output / 'STOP').exists():
                    raise RuntimeError('owner STOP')
                available = guard.owner.host_available()
                if available < 96 * 2**30:
                    raise RuntimeError('host available below 96 GiB')
                gpu = guard.gpu_state()
                if min(gpu[rank]['free_mib'] for rank in PHYSICAL) < 2048:
                    raise RuntimeError('R4 GPU free memory below 2 GiB')
                if max(gpu[rank]['temperature_c'] for rank in PHYSICAL) >= 85:
                    raise RuntimeError('R4 GPU temperature reached 85 C')
                phase = json.loads((output / 'phase.json').read_text()) if (output / 'phase.json').exists() else {'phase': 'loading'}
                resources.write(json.dumps(dict(unix=time.time(), phase=phase,
                                                gpus=gpu, host_available=available)) + '\n')
                resources.flush()
                time.sleep(1)
        state['worker_returncode'] = process.returncode
        assert process.returncode == 0, f'worker exited {process.returncode}; see run.log'
        result = json.loads((output / 'result.json').read_text())
        assert result['status'] == 'PASS'
        if args.diagnostic:
            for rank in range(4):
                assert json.loads((output / f'generation_diagnostic_rank{rank}.json').read_text())['status'] == 'PASS'
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
        while any(gpu in range(8) for gpu, _ in guard.apps()) and time.monotonic() < deadline:
            time.sleep(1)
        state['remaining_gpu_processes'] = guard.apps()
        state['restored_model_loads'] = guard.restore_owned_loads(stopped)
        state['finished'] = time.time()
        guard.write(output / 'status.json', state)


if __name__ == '__main__':
    main()
