"""Guarded C20/C50 Qwen fetch diagnostics, after the clean primary sweep."""

import argparse
import json
import os
import subprocess
from pathlib import Path


PKG = Path(__file__).resolve().parents[1]
SCRIPT = PKG / 'scripts/run_headline_job.py'
MANIFEST = Path('/home/hwlee/mgo-results/qwen_cache_ablation_20261009/WORKLOADS.json')
JOBS = Path('/home/hwlee/mgo-results/headline_r4_20261007')
PRIMARY = PKG / 'experiments/qwen_cache_ablation_20261009/PROGRESS.json'
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    progress = json.loads(PRIMARY.read_text())
    assert progress['status'] == 'PASS' and progress['completed'] == 4
    manifest = json.loads(MANIFEST.read_text())
    assert manifest['status'] == 'FROZEN' and manifest['physical_gpus'] == [0, 1, 4, 5]
    for percent in (20, 50):
        cell = next(c for c in manifest['cells'] if c['cache_percent'] == percent)
        assert cell['model'] == 'Qwen3' and cell['local_batch'] == 16
        prefix = f'qca_fetch_c{percent}_b16_r1_v'
        paths = sorted(JOBS.glob(prefix + '*'),
                       key=lambda path: int(path.name.rsplit('_v', 1)[1]))
        passed = [path for path in paths if (path / 'status.json').exists()
                  and json.loads((path / 'status.json').read_text()).get('status') == 'PASS'
                  and all((path / f'generation_diagnostic_rank{r}.json').exists()
                          for r in range(4))]
        if passed:
            print('REUSE PASS', passed[-1].name, flush=True)
            continue
        number = max((int(path.name.rsplit('_v', 1)[1]) for path in paths), default=0) + 1
        label = prefix + str(number)
        command = [PYTHON, '-u', str(SCRIPT), '--workloads', str(MANIFEST),
                   '--label', label, '--cell', cell['cell'], '--repeats', '1',
                   '--system', 'Ours-diagnostic', '--worker', 'headline_ours_worker.py',
                   '--ours-final', '--post-generation-diagnostic', '--timeout', '7200']
        print('RUN', label, flush=True)
        if args.dry_run:
            print(json.dumps(command), flush=True)
            continue
        env = dict(os.environ, MGO_MIN_GPU_FREE_MIB='2048')
        subprocess.run(command, check=True, env=env)
        status = json.loads((JOBS / label / 'status.json').read_text())
        assert status['status'] == 'PASS'
        assert all((JOBS / label / f'generation_diagnostic_rank{r}.json').exists()
                   for r in range(4))
        print('PASS', label, flush=True)


if __name__ == '__main__':
    main()
