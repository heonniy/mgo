"""Interleaved stage-breakdown diagnostics: 5 arms x 2 models x 4 repetitions (serial + inline H2D)."""
import json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
PY = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TABLE = PKG / 'experiments/pcie_quota_r8_b16_20261010/LOOKUP.json'
DS_TEACHER = PKG / 'experiments/deepseek_r8_sharegpt_b16_l512_20261012/FROZEN_TOKENS.json'
ARMS = {'near': 'LA_CA_NEAR', 'fast': 'NEAR_FAST', 'rand': 'FAST_RANDOM', 'worst': 'FAST_WORST', 'haqf': 'HAQ_FAST'}
TABLE_POLICIES = ('NEAR_FAST', 'FAST_RANDOM', 'FAST_WORST')
QROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
DROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012/jobs')
N = list(ARMS)
ORDERS = (N, N[::-1], N[2:] + N[:2], (N[2:] + N[:2])[::-1])
REPS = len(ORDERS)


def write(data):
    p = HERE / 'STATUS.json'; t = p.with_suffix('.json.tmp'); t.write_text(json.dumps(data, indent=2) + '\n'); t.replace(p)


def qwen(tag, rep):
    out = QROOT / f'sbq_{tag}_full_v{rep}'
    if out.exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'PASS'; return out
    policy = ARMS[tag]
    cmd = [PY, '-u', str(PKG / 'scripts/run_qwen_r8_job.py'), '--system', 'ours', '--ours-mode', 'A',
           '--job-label', f'sbq_{tag}', '--attempt', str(rep), '--ours-policy', policy, '--repeats', '1',
           '--quiet-2367', '--inline-demand-h2d', '--h2d-serial-ablation', '--post-generation-diagnostic']
    if policy in TABLE_POLICIES:
        cmd += ['--quota-table', str(TABLE)]
    subprocess.run(cmd, check=True, cwd=PKG.parent)
    return out


def deepseek(tag, rep):
    out = DROOT / f'sbd_{tag}_full_v{rep}'
    if out.exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'PASS'; return out
    policy = ARMS[tag]
    cmd = [PY, '-u', str(PKG / 'scripts/run_deepseek_r8_job.py'), '--decode-policy', policy, '--repeats', '1',
           '--attempt', str(rep), '--inline-demand-h2d', '--h2d-serial-ablation',
           '--teacher-tokens', str(DS_TEACHER), '--job-label', f'sbd_{tag}']
    if policy in TABLE_POLICIES:
        cmd += ['--quota-table', str(TABLE)]
    subprocess.run(cmd, check=True, cwd=PKG.parent, env=dict(os.environ, MGO_DEEPSEEK_CACHE_DIAG='1'))
    return out


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status)
    try:
        for rep, order in enumerate(ORDERS, 1):
            for model, fn in (('qwen', qwen), ('deepseek', deepseek)):
                for tag in order:
                    assert not (HERE / 'STOP').exists()
                    out = fn(tag, rep)
                    status['jobs'].append(dict(model=model, arm=tag, policy=ARMS[tag], rep=rep, output=str(out)))
                    write(status)
                    print('PASS', model, tag, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
