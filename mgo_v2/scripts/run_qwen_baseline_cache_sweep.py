"""Resume the guarded Qwen C20/C40/C50 baseline sweep."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


P = Path(__file__).resolve().parents[1]
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SUPERVISOR = P / 'scripts/run_headline_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE = '/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
INFINITY = '/home/hwlee/mgo-tools/headline-r4/infinity-env/bin/python'
SYSTEMS = ('infinity', 'deepspeed', 'llama')
BUILD = P / 'experiments/main_table_global_workload_20261006/expanded_matrix/LLAMA_BUILD.json'
LLAMA_SOURCE = P / 'examples/headline_llama_sync.cpp'


def ensure_llama_build():
    receipt = json.loads(BUILD.read_text())
    source_sha = hashlib.sha256(LLAMA_SOURCE.read_bytes()).hexdigest()
    if receipt['source_sha256'] != source_sha:
        print('BUILD balanced Qwen llama executable between guarded jobs', flush=True)
        subprocess.run([PYTHON, '-u', str(P / 'scripts/build_headline_llama_sync.py')], check=True)
    assert json.loads(BUILD.read_text())['source_sha256'] == source_sha


def wait_on_running(paths):
    active = [p for p in paths if (p / 'status.json').exists() and
              json.loads((p / 'status.json').read_text()).get('status') == 'RUNNING']
    if not active:
        return
    assert len(active) == 1, active
    path = active[0]
    print('WAIT current guarded attempt', path.name, flush=True)
    deadline = time.monotonic() + 7400
    while time.monotonic() < deadline:
        status = json.loads((path / 'status.json').read_text())
        if status['status'] != 'RUNNING':
            print('FINISHED', path.name, status['status'], flush=True)
            return
        time.sleep(10)
    raise TimeoutError(f'guarded attempt did not finalize: {path}')


def command(cell, system, label, repeats):
    common = [PYTHON, '-u', str(SUPERVISOR), '--workloads', str(MANIFEST),
              '--label', label, '--cell', cell['cell'], '--repeats', str(repeats)]
    if system == 'infinity':
        return common + ['--system', 'MoE-Infinity-repaired', '--worker',
                         'headline_infinity_worker.py', '--python', INFINITY,
                         '--ranks', '1', '--timeout', '7200']
    if system == 'deepspeed':
        return common + ['--system', 'DeepSpeed-ZeRO-Inference', '--worker',
                         'headline_deepspeed_worker.py', '--python', BASE,
                         '--timeout', '7200']
    if system == 'llama':
        per = {20: 2, 30: 3, 40: 4, 50: 6}[cell['cache_percent']]
        return common + ['--system', f'llama.cpp-sync-balanced{per}', '--worker',
                         'headline_llama_sync_worker.py', '--python', BASE,
                         '--ranks', '1', '--llama-threads', '32',
                         '--llama-cuda-graphs', 'off', '--llama-graph-reuse', 'off',
                         '--llama-expert-placement', f'balanced{per}',
                         '--timeout', '28800']
    raise ValueError(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', choices=SYSTEMS, action='append')
    parser.add_argument('--percent', type=int, choices=(20, 40, 50), action='append')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == [0, 1, 4, 5]
    assert len(manifest['cells']) == 4
    for cell in sorted(manifest['cells'], key=lambda item: item['cache_percent']):
        cap = cell['cache_percent']
        if cap == 30 or (args.percent and cap not in args.percent):
            continue  # The matching three-repeat C30 records already exist.
        assert cell['model'] == 'Qwen3' and cell['dataset'] == 'ShareGPT'
        assert cell['local_batch'] == 16 and cell['input_tokens'] == 512
        assert cell['output_tokens'] == 64
        for system in args.system or SYSTEMS:
            prefix = f'qca_baseline_c{cap}_{system}_'
            attempts = sorted(JOBS.glob(prefix + '*'))
            if not args.dry_run:
                wait_on_running(attempts)
            primary_attempts = [p for p in attempts if '_r2_v' in p.name or '_r3_v' in p.name]
            passed = [p for p in primary_attempts if (p / 'status.json').exists() and
                      json.loads((p / 'status.json').read_text()).get('status') == 'PASS']
            if passed:
                primary = passed[-1]
                print('REUSE PASS', primary.name, flush=True)
            else:
                primary_prefix = prefix + 'r2_v'
                number = max((int(p.name.rsplit('_v', 1)[1]) for p in primary_attempts
                              if '_r2_v' in p.name), default=0) + 1
                label = primary_prefix + str(number)
                call = command(cell, system, label, 2)
                print('RUN', label, flush=True)
                if args.dry_run:
                    print(json.dumps(call), flush=True)
                    continue
                if system == 'llama':
                    ensure_llama_build()
                env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
                subprocess.run(call, check=True, env=env)
                primary = JOBS / label
                assert json.loads((primary / 'status.json').read_text())['status'] == 'PASS'
                print('PASS', label, flush=True)
            if '_r3_v' in primary.name or args.dry_run:
                continue
            measurements = [json.loads((primary / f'repeat{i}.json').read_text()) for i in (1, 2)]
            differences = {metric: abs(measurements[0][metric] - measurements[1][metric]) /
                           ((measurements[0][metric] + measurements[1][metric]) / 2)
                           for metric in ('TPOT', 'E2E')}
            maximum = max(differences.values())
            if maximum <= .02:
                print('STABLE TWO', primary.name, differences, flush=True)
                continue
            if maximum > .05:
                print('UNSTABLE >5%; NO MORE REPEATS', primary.name, differences, flush=True)
                continue
            follow_prefix = prefix + 'followup_r1_v'
            followups = sorted(JOBS.glob(follow_prefix + '*'))
            if any((p / 'status.json').exists() and
                   json.loads((p / 'status.json').read_text()).get('status') == 'PASS'
                   for p in followups):
                print('REUSE THIRD', followups[-1].name, flush=True)
                continue
            label = follow_prefix + str(max((int(p.name.rsplit('_v', 1)[1])
                                             for p in followups), default=0) + 1)
            print('RUN THIRD', label, differences, flush=True)
            if system == 'llama':
                ensure_llama_build()
            env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
            subprocess.run(command(cell, system, label, 1), check=True, env=env)
            assert json.loads((JOBS / label / 'status.json').read_text())['status'] == 'PASS'


if __name__ == '__main__':
    main()
