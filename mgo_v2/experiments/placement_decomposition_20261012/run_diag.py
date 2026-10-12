"""Run the six stage-2 decomposition diagnostic jobs (Qwen + DeepSeek, FAST quota, serial + inline)."""
import json, os, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
PY = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TABLE = PKG / 'experiments/pcie_quota_r8_b16_20261010/LOOKUP.json'
DS_TEACHER = PKG / 'experiments/deepseek_r8_sharegpt_b16_l512_20261012/FROZEN_TOKENS.json'
ARMS = {'near': 'NEAR_FAST', 'rand': 'FAST_RANDOM', 'worst': 'FAST_WORST'}
QROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
DROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012/jobs')

for tag, policy in ARMS.items():
    out = QROOT / f'qsd_{tag}_full_v1'
    if not out.exists():
        subprocess.run([PY, '-u', str(PKG / 'scripts/run_qwen_r8_job.py'), '--system', 'ours', '--ours-mode', 'A',
                        '--job-label', f'qsd_{tag}', '--ours-policy', policy, '--repeats', '1', '--quiet-2367',
                        '--inline-demand-h2d', '--h2d-serial-ablation', '--post-generation-diagnostic',
                        '--quota-table', str(TABLE)], check=True, cwd=PKG.parent)
        print('PASS', out.name, flush=True)
    out = DROOT / f'dsd_{tag}_full_v1'
    if not out.exists():
        subprocess.run([PY, '-u', str(PKG / 'scripts/run_deepseek_r8_job.py'), '--decode-policy', policy,
                        '--repeats', '1', '--inline-demand-h2d', '--h2d-serial-ablation',
                        '--teacher-tokens', str(DS_TEACHER), '--quota-table', str(TABLE),
                        '--job-label', f'dsd_{tag}'], check=True, cwd=PKG.parent,
                       env=dict(os.environ, MGO_DEEPSEEK_CACHE_DIAG='1'))
        print('PASS', out.name, flush=True)
