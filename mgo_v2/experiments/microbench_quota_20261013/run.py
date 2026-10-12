"""Qwen R8 microbench-quota campaign (no hogs; serial + inline H2D; Near placement; post-generation diagnostic).

Arms: NEAR, FAST, MB_BAL (microbench-best balanced quota), MB_OPT (microbench-best unconstrained quota),
BRAND_W (balanced random quota with the model-worst salt). 3 reps, arm order rotated per rep.
Unconstrained random quota (RANDOM_QUOTA_NEAR) is reused from stage_breakdown_20261012.
"""
import json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
PY = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
ROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
WORST_SALT = json.loads((HERE / 'WORST_SALT.json').read_text())['worst_salt']
ARMS = {'near': ('LA_CA_NEAR', None, None),
        'fast': ('NEAR_FAST', PKG / 'experiments/pcie_quota_r8_b16_20261010/LOOKUP.json', None),
        'mbbal': ('NEAR_SPLIT', HERE / 'MB_BAL_qwen.json', None),
        'mbopt': ('NEAR_SPLIT', HERE / 'MB_OPT_qwen.json', None),
        'bw': ('BALANCED_RANDOM_NEAR', None, WORST_SALT)}
ORDER = list(ARMS)[::-1]  # MB_OPT/BRAND_W first: fail fast on the new paths


def write(d):
    p = HERE / 'STATUS.json'; t = p.with_name('STATUS.json.tmp'); t.write_text(json.dumps(d, indent=2) + '\n'); t.replace(p)


def job(arm, rep):
    policy, table, salt = ARMS[arm]
    label = f'mbq{arm}'; out = ROOT / f'{label}_full_v{rep}'
    if out.exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'PASS'; return out
    cmd = [PY, '-u', str(PKG / 'scripts/run_qwen_r8_job.py'), '--system', 'ours', '--ours-mode', 'A',
           '--job-label', label, '--attempt', str(rep), '--ours-policy', policy, '--repeats', '1',
           '--quiet-2367', '--inline-demand-h2d', '--h2d-serial-ablation', '--post-generation-diagnostic']
    if table is not None:
        cmd += ['--quota-table', str(table)]
    env = dict(os.environ)
    if salt is not None:
        env['MGO_RANDOM_SALT'] = str(salt)
    subprocess.run(cmd, check=True, cwd=PKG.parent, env=env)
    return out


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'worst_salt': WORST_SALT, 'jobs': []}
    write(status)
    try:
        for rep in (1, 2, 3):
            k = (rep - 1) % len(ORDER)
            for arm in ORDER[k:] + ORDER[:k]:
                assert not (HERE / 'STOP').exists()
                out = job(arm, rep)
                status['jobs'].append(dict(model='qwen', arm=arm, policy=ARMS[arm][0], rep=rep, output=str(out)))
                write(status); print('PASS', arm, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
