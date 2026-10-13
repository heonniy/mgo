"""Qwen-only follow-up at the 1.3x / 1.7x hog levels: MB_BAL, balanced random with the worst salt (406),
and balanced random with ordinary salts (salt = rep, 1/2/3). 3 reps; arm order rotated per rep.
Reuses icql13brand_full_v1 (salt 1, campaign 2) as BRAND rep 1 at 1.3x. NEAR/FAST/SPLIT come from campaign 1.
MB_BAL is the no-hog microbench table (microbench_quota_20261013/MB_BAL_qwen.json), used unchanged.
"""
import json, os, subprocess, time
from run import HERE, PKG, PY, QROOT, LEVELS, write

MB_BAL = PKG / 'experiments/microbench_quota_20261013/MB_BAL_qwen.json'
WORST = json.loads((PKG / 'experiments/microbench_quota_20261013/CONTENTION_SALTS.json').read_text())
ARMS = {'mbbal': ('NEAR_SPLIT', MB_BAL, None), 'bw': ('BALANCED_RANDOM_NEAR', None, 'worst'),
        'brand': ('BALANCED_RANDOM_NEAR', None, 'rep')}
ORDER = ['mbbal', 'bw', 'brand']


def job(level, arm, rep):
    policy, table, salt = ARMS[arm]
    salt = WORST[level]['worst_salt'] if salt == 'worst' else rep if salt == 'rep' else None
    label = f'icq{level}{arm}'; out = QROOT / f'{label}_full_v{rep}'
    if out.exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'PASS'; return out, salt
    cmd = [PY, '-u', str(PKG / 'scripts/run_qwen_r8_job.py'), '--system', 'ours', '--ours-mode', 'A',
           '--job-label', label, '--attempt', str(rep), '--ours-policy', policy, '--repeats', '1',
           '--quiet-2367', '--inline-demand-h2d', '--h2d-serial-ablation', '--post-generation-diagnostic',
           '--hog-gpus', '0,1,2,3', '--hog-streams', str(LEVELS[level]['streams']), '--hog-gibps', '0']
    if table is not None:
        cmd += ['--quota-table', str(table)]
    env = dict(os.environ)
    if salt is not None:
        env['MGO_RANDOM_SALT'] = str(salt)
    subprocess.run(cmd, check=True, cwd=PKG.parent, env=env)
    return out, salt


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status, 'STATUS3.json')
    try:
        for rep in (1, 2, 3):
            arms = ORDER[rep - 1:] + ORDER[:rep - 1]
            for level in ('l13', 'l17'):
                for arm in arms:
                    assert not (HERE / 'STOP').exists()
                    out, salt = job(level, arm, rep)
                    status['jobs'].append(dict(model='qwen', level=level, arm=arm, rep=rep, salt=salt, output=str(out)))
                    write(status, 'STATUS3.json'); print('PASS', level, arm, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status, 'STATUS3.json')


if __name__ == '__main__':
    main()
