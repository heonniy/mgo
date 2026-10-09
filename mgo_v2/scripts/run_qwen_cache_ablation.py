"""Run and resume four guarded Qwen main_OURS cache-capacity jobs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
SCRIPT = PKG / 'scripts/run_headline_job.py'
REPORTER = PKG / 'scripts/report_qwen_cache_ablation.py'
REPORT = PKG / 'experiments/qwen_cache_ablation_20261009/PROGRESS.json'
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'


def attempts(prefix):
    return sorted(JOBS.glob(prefix + '*'),
                  key=lambda path: int(path.name.rsplit('_v', 1)[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--percent', type=int, choices=(20, 30, 40, 50), action='append')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == [0, 1, 4, 5]
    assert sorted(c['cache_percent'] for c in manifest['cells']) == [20, 30, 40, 50]
    for cell in sorted(manifest['cells'], key=lambda value: value['cache_percent']):
        if args.percent and cell['cache_percent'] not in args.percent:
            continue
        assert cell['model'] == 'Qwen3' and cell['local_batch'] == 16
        assert cell['input_tokens'] == 512 and cell['output_tokens'] == 64
        prefix = f'qca_c{cell["cache_percent"]}_b16_ours_r3_v'
        prior = attempts(prefix)
        passed = [path for path in prior if (path / 'status.json').exists()
                  and json.loads((path / 'status.json').read_text()).get('status') == 'PASS']
        if passed:
            print('REUSE PASS', passed[-1].name, flush=True)
        else:
            number = max((int(path.name.rsplit('_v', 1)[1]) for path in prior), default=0) + 1
            label = prefix + str(number)
            command = [PYTHON, '-u', str(SCRIPT), '--workloads', str(MANIFEST),
                       '--label', label, '--cell', cell['cell'], '--repeats', '3',
                       '--system', 'Ours', '--worker', 'headline_ours_worker.py',
                       '--ours-final', '--timeout', '7200']
            print('RUN', label, flush=True)
            if args.dry_run:
                print(json.dumps(command), flush=True)
                continue
            env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
            subprocess.run(command, check=True, env=env)
            assert json.loads((JOBS / label / 'status.json').read_text())['status'] == 'PASS'
        if args.dry_run:
            continue
        subprocess.run([sys.executable, str(REPORTER)], check=True)
        progress = json.loads(REPORT.read_text())
        assert next(c for c in progress['cases'] if c['cache_percent'] ==
                    cell['cache_percent'])['status'] == 'PASS'
        print('PASS', cell['cell'], flush=True)


if __name__ == '__main__':
    main()
