"""Untimed post-generation diagnostic pass for NEAR, FAST and NEAR_SPLIT (live tokens, 1 target)."""

import json
import subprocess
from pathlib import Path

from run_split import GUARD, PKG, PYTHON, ROOT, TABLES

LABELS = {'LA_CA_NEAR': 'diag_near', 'NEAR_FAST': 'diag_fast', 'NEAR_SPLIT': 'diag_split'}

for policy, label in LABELS.items():
    output = ROOT / 'jobs' / f'{label}_full_v1'
    if output.exists() and json.loads((output / 'status.json').read_text())['status'] == 'PASS':
        continue
    command = [PYTHON, '-u', str(GUARD), '--system', 'ours', '--ours-mode', 'N',
               '--job-label', label, '--ours-policy', policy, '--attempt', '1',
               '--repeats', '1', '--quiet-2367', '--post-generation-diagnostic']
    if TABLES[policy] is not None:
        command += ['--quota-table', str(TABLES[policy])]
    subprocess.run(command, check=True, cwd=PKG.parent)
    print('PASS', output.name, flush=True)
