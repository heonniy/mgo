"""Resume the guarded Qwen C20/C40/C50 baseline sweep."""

import argparse
import json
import os
import subprocess
from pathlib import Path


P = Path(__file__).resolve().parents[1]
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SUPERVISOR = P / 'scripts/run_headline_job.py'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE = '/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
INFINITY = '/home/hwlee/mgo-tools/headline-r4/infinity-env/bin/python'
SYSTEMS = ('infinity', 'deepspeed', 'llama')


def command(cell, system, label):
    common = [PYTHON, '-u', str(SUPERVISOR), '--workloads', str(MANIFEST),
              '--label', label, '--cell', cell['cell'], '--repeats', '3']
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
            prefix = f'qca_baseline_c{cap}_{system}_r3_v'
            attempts = sorted(JOBS.glob(prefix + '*'))
            passed = [p for p in attempts if (p / 'status.json').exists() and
                      json.loads((p / 'status.json').read_text()).get('status') == 'PASS']
            if passed:
                print('REUSE PASS', passed[-1].name, flush=True)
                continue
            number = max((int(p.name.rsplit('_v', 1)[1]) for p in attempts), default=0) + 1
            label = prefix + str(number)
            call = command(cell, system, label)
            print('RUN', label, flush=True)
            if args.dry_run:
                print(json.dumps(call), flush=True)
                continue
            env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
            subprocess.run(call, check=True, env=env)
            assert json.loads((JOBS / label / 'status.json').read_text())['status'] == 'PASS'
            print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
