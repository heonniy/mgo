"""Resume the 32 DeepSeek headline jobs, one guarded owner-GPU job at a time."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_main_table_2x2_qwen import (SCRIPT, ROOT, WORKLOAD_ROOT, OURS_PYTHON,
                                      BASE_PYTHON, INFINITY_PYTHON, SYSTEMS,
                                      existing_pass, next_label)


def command(dataset, cell, system, label, smoke=False):
    common = [OURS_PYTHON, '-u', str(SCRIPT), '--workloads',
              str(WORKLOAD_ROOT / dataset / 'WORKLOADS.json'), '--label', label,
              '--cell', cell, '--repeats', '3']
    if smoke:
        common.append('--smoke')
    if system == 'ours':
        return common + ['--system', 'Ours', '--worker',
                         'headline_ours_deepseek_worker.py', '--timeout', '7200']
    if system == 'deepspeed':
        return common + ['--system', 'DeepSpeed-ZeRO-Inference', '--worker',
                         'headline_deepspeed_worker.py', '--python', BASE_PYTHON,
                         '--timeout', '7200']
    if system == 'infinity':
        return common + ['--system', 'MoE-Infinity-repaired', '--worker',
                         'headline_infinity_worker.py', '--python', INFINITY_PYTHON,
                         '--ranks', '1', '--timeout', '7200']
    if system == 'llama':
        return common + ['--system', 'llama.cpp-sync-balanced4', '--worker',
                         'headline_llama_deepseek_sync_worker.py', '--python', BASE_PYTHON,
                         '--ranks', '1', '--llama-threads', '32',
                         '--llama-cuda-graphs', 'off', '--llama-graph-reuse', 'off',
                         '--llama-expert-placement', 'balanced4', '--timeout', '28800']
    raise ValueError(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--system', choices=SYSTEMS, action='append')
    parser.add_argument('--dataset', choices=('ShareGPT', 'LMSYS-Chat-1M'), action='append')
    args = parser.parse_args()
    for dataset in args.dataset or ('ShareGPT', 'LMSYS-Chat-1M'):
        manifest = json.loads((WORKLOAD_ROOT / dataset / 'WORKLOADS.json').read_text())
        assert manifest['status'] == 'FROZEN'
        for cell in manifest['cells']:
            if cell['model'] != 'DeepSeekV2Lite':
                continue
            for system in args.system or SYSTEMS:
                suffix = '_smoke' if args.smoke else '_r3'
                prefix = (f'mt2_deepseek_{dataset.lower().replace("-", "_")}'
                          f'_b{cell["local_batch"]}_l{cell["input_tokens"]}_{system}{suffix}')
                passed = existing_pass(prefix)
                if passed:
                    print('REUSE PASS', passed.name, flush=True)
                    continue
                label = next_label(prefix)
                call = command(dataset, cell['cell'], system, label, args.smoke)
                print('RUN', label, flush=True)
                if args.dry_run:
                    print(json.dumps(call), flush=True)
                    continue
                subprocess.run(call, check=True)
                status = json.loads((ROOT / label / 'status.json').read_text())
                assert status['status'] == 'PASS', label
                if not args.smoke:
                    report = Path(__file__).with_name('report_main_table_2x2.py')
                    subprocess.run([sys.executable, str(report)], check=True)
                    progress_path = (Path(__file__).resolve().parents[1] /
                                     'experiments/main_table_2x2_20261008/PROGRESS.json')
                    progress = json.loads(progress_path.read_text())
                    assert any(case.get('selected_attempt') == label and case['status'] == 'PASS'
                               for case in progress['cases']), label
                print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
