"""Run the B16 DeepSeek cache sweep, one guarded owner-GPU job at a time."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
SCRIPT = PKG / 'scripts/run_headline_job.py'
REPORT_SCRIPT = PKG / 'scripts/report_deepseek_cache_ablation.py'
REPORT = PKG / 'experiments/deepseek_cache_ablation_20261009/PROGRESS.json'
MANIFEST = Path('/home/hwlee/mgo-results/deepseek_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
OURS_PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE_PYTHON = '/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
INFINITY_PYTHON = '/home/hwlee/mgo-tools/headline-r4/infinity-env/bin/python'
SYSTEMS = ('ours', 'infinity', 'deepspeed', 'llama')


def label_prefix(cell, system):
    return f'dca_c{cell["cache_percent"]}_b{cell["local_batch"]}_{system}_r3_v'


def attempts(prefix):
    return sorted(JOBS.glob(prefix + '*'), key=lambda path: int(path.name.rsplit('_v', 1)[1]))


def command(cell, system, label):
    common = [OURS_PYTHON, '-u', str(SCRIPT), '--workloads', str(MANIFEST),
              '--label', label, '--cell', cell['cell'], '--repeats', '3']
    if system == 'ours':
        return common + ['--system', 'Ours', '--worker',
                         'headline_ours_deepseek_worker.py', '--timeout', '7200']
    if system == 'infinity':
        return common + ['--system', 'MoE-Infinity-repaired', '--worker',
                         'headline_infinity_worker.py', '--python', INFINITY_PYTHON,
                         '--ranks', '1', '--timeout', '7200']
    if system == 'deepspeed':
        return common + ['--system', 'DeepSpeed-ZeRO-Inference', '--worker',
                         'headline_deepspeed_worker.py', '--python', BASE_PYTHON,
                         '--timeout', '7200']
    if system == 'llama':
        layers = 4 * (min(cell['expert_slots_per_rank']) // 64)
        return common + ['--system', f'llama.cpp-sync-balanced{layers}', '--worker',
                         'headline_llama_deepseek_sync_worker.py', '--python', BASE_PYTHON,
                         '--ranks', '1', '--llama-threads', '32',
                         '--llama-cuda-graphs', 'off', '--llama-graph-reuse', 'off',
                         '--llama-expert-placement', f'balanced{layers}',
                         '--timeout', '28800']
    raise ValueError(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--percent', type=int, choices=(20, 30, 40, 50), action='append')
    parser.add_argument('--system', choices=SYSTEMS, action='append')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == [0, 1, 4, 5]
    assert len(manifest['cells']) == 4
    assert all(cell['model'] == 'DeepSeekV2Lite' and cell['dataset'] == 'ShareGPT'
               and cell['local_batch'] == 16 and cell['input_tokens'] == 512
               and cell['output_tokens'] == 64 for cell in manifest['cells'])
    for cell in sorted(manifest['cells'], key=lambda value: value['cache_percent']):
        if args.percent and cell['cache_percent'] not in args.percent:
            continue
        for system in args.system or SYSTEMS:
            prefix = label_prefix(cell, system)
            prior = attempts(prefix)
            passed = [path for path in prior if (path / 'status.json').exists()
                      and json.loads((path / 'status.json').read_text()).get('status') == 'PASS']
            if passed:
                print('REUSE PASS', passed[-1].name, flush=True)
            else:
                label = prefix + str(max((int(path.name.rsplit('_v', 1)[1])
                                          for path in prior), default=0) + 1)
                call = command(cell, system, label)
                print('RUN', label, flush=True)
                if args.dry_run:
                    print(json.dumps(call), flush=True)
                    continue
                env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
                subprocess.run(call, check=True, env=env)
                status = json.loads((JOBS / label / 'status.json').read_text())
                assert status['status'] == 'PASS', label
            if args.dry_run:
                continue
            subprocess.run([sys.executable, str(REPORT_SCRIPT)], check=True)
            progress = json.loads(REPORT.read_text())
            assert any(case['cell'] == cell['cell'] and case['system'] == system
                       and case['status'] == 'PASS' for case in progress['cases'])
            print('PASS', cell['cell'], system, flush=True)


if __name__ == '__main__':
    main()
