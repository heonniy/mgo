"""Untimed post-generation diagnostic pass (live tokens) for the three FAST stage-2 arms, inline H2D."""

import json
import subprocess

from run_stage2 import FAST_TABLE, GUARD, PKG, PYTHON, ROOT

ARMS = {'d2_fastnear': 'NEAR_FAST', 'd2_fastrand': 'FAST_RANDOM', 'd2_fastworst': 'FAST_WORST'}

for label, policy in ARMS.items():
    output = ROOT / 'jobs' / f'{label}_full_v1'
    if output.exists() and json.loads((output / 'status.json').read_text())['status'] == 'PASS':
        continue
    subprocess.run([PYTHON, '-u', str(GUARD), '--system', 'ours', '--ours-mode', 'N',
                    '--job-label', label, '--ours-policy', policy, '--attempt', '1',
                    '--repeats', '1', '--quiet-2367', '--inline-demand-h2d',
                    '--post-generation-diagnostic', '--quota-table', str(FAST_TABLE)],
                   check=True, cwd=PKG.parent)
    print('PASS', output.name, flush=True)
