"""Resume the 32 Qwen headline jobs, one guarded four-GPU job at a time."""
import argparse
import json
import subprocess
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
SCRIPT = PKG / 'scripts/run_headline_job.py'
ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
WORKLOAD_ROOT = Path('/home/hwlee/mgo-results/main_table_2x2_20261008')
OURS_PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE_PYTHON = '/home/hwlee/mgo-tools/headline-r4/base-env/bin/python'
INFINITY_PYTHON = '/home/hwlee/mgo-tools/headline-r4/infinity-env/bin/python'
SYSTEMS = ('ours', 'deepspeed', 'infinity', 'llama')


def command(dataset, cell, system, label):
    common = [OURS_PYTHON, '-u', str(SCRIPT), '--workloads', str(WORKLOAD_ROOT / dataset / 'WORKLOADS.json'),
              '--label', label, '--cell', cell, '--repeats', '3']
    if system == 'ours':
        return common + ['--system', 'Ours', '--worker', 'headline_ours_worker.py', '--ours-final', '--timeout', '7200']
    if system == 'deepspeed':
        return common + ['--system', 'DeepSpeed-ZeRO-Inference', '--worker', 'headline_deepspeed_worker.py',
                         '--python', BASE_PYTHON, '--timeout', '7200']
    if system == 'infinity':
        return common + ['--system', 'MoE-Infinity-repaired', '--worker', 'headline_infinity_worker.py',
                         '--python', INFINITY_PYTHON, '--ranks', '1', '--timeout', '7200']
    if system == 'llama':
        return common + ['--system', 'llama.cpp-sync-balanced3', '--worker', 'headline_llama_sync_worker.py',
                         '--python', BASE_PYTHON, '--ranks', '1', '--llama-threads', '32',
                         '--llama-cuda-graphs', 'off', '--llama-graph-reuse', 'off',
                         '--llama-expert-placement', 'balanced3', '--timeout', '28800']
    raise ValueError(system)


def existing_pass(prefix):
    for path in sorted(ROOT.glob(prefix + '_v*'), reverse=True):
        status = path / 'status.json'
        if status.exists() and json.loads(status.read_text()).get('status') == 'PASS':
            return path
    return None


def next_label(prefix):
    for attempt in range(1, 100):
        label = f'{prefix}_v{attempt}'
        if not (ROOT / label).exists():
            return label
    raise RuntimeError(f'too many attempts for {prefix}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--system', choices=SYSTEMS, action='append')
    parser.add_argument('--dataset', choices=('ShareGPT', 'LMSYS-Chat-1M'), action='append')
    args = parser.parse_args()
    for dataset in args.dataset or ('ShareGPT', 'LMSYS-Chat-1M'):
        manifest = json.loads((WORKLOAD_ROOT / dataset / 'WORKLOADS.json').read_text())
        assert manifest['status'] == 'FROZEN'
        for cell in manifest['cells']:
            if cell['model'] != 'Qwen3':
                continue
            for system in args.system or SYSTEMS:
                prefix = f'mt2_qwen_{dataset.lower().replace("-", "_")}_b{cell["local_batch"]}_l{cell["input_tokens"]}_{system}_r3'
                passed = existing_pass(prefix)
                if passed:
                    print('REUSE PASS', passed.name, flush=True)
                    continue
                label = next_label(prefix)
                call = command(dataset, cell['cell'], system, label)
                print('RUN', label, flush=True)
                if args.dry_run:
                    print(json.dumps(call), flush=True)
                    continue
                subprocess.run(call, check=True)
                status = json.loads((ROOT / label / 'status.json').read_text())
                assert status['status'] == 'PASS', label
                print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
