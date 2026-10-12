"""Induced-contention campaign: 2 levels x 2 models x {NEAR, FAST, SPLIT} x 3 reps."""
import json, os, subprocess, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
PY = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
FAST_TABLE = PKG / 'experiments/pcie_quota_r8_b16_20261010/LOOKUP.json'
DS_TEACHER = PKG / 'experiments/deepseek_r8_sharegpt_b16_l512_20261012/FROZEN_TOKENS.json'
LEVELS = {'l00': dict(streams=0, table=HERE / 'SPLIT_r144.json'), 'l13': dict(streams=1, table=HERE / 'SPLIT_r129.json'), 'l17': dict(streams=2, table=HERE / 'SPLIT_r172.json')}
ARMS = {'near': 'LA_CA_NEAR', 'fast': 'NEAR_FAST', 'split': 'NEAR_SPLIT', 'brand': 'BALANCED_RANDOM_NEAR'}
ORDERS = (('near', 'fast', 'split'), ('split', 'fast', 'near'), ('fast', 'split', 'near'))
QROOT = Path('/home/hwlee/mgo-results/qwen_r8_sharegpt_b16_l512_20261009/jobs')
DROOT = Path('/home/hwlee/mgo-results/deepseek_r8_sharegpt_b16_l512_20261012/jobs')


def write(d, name='STATUS.json'):
    p = HERE / name; t = p.with_name(p.name + '.tmp'); t.write_text(json.dumps(d, indent=2) + '\n'); t.replace(p)


def job(model, level, arm, rep, smoke=False):
    policy = ARMS[arm]; lv = LEVELS[level]
    label = f"ic{'q' if model == 'qwen' else 'd'}{level}{arm}"
    root = QROOT if model == 'qwen' else DROOT
    out = root / f"{label}_{'smoke' if smoke else 'full'}_v{rep}"
    if out.exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'PASS'; return out
    hog = ['--hog-gpus', '0,1,2,3', '--hog-streams', str(lv['streams']), '--hog-gibps', '0'] if lv['streams'] else []
    table = {'NEAR_FAST': FAST_TABLE, 'NEAR_SPLIT': lv['table']}.get(policy)
    if model == 'qwen':
        cmd = [PY, '-u', str(PKG / 'scripts/run_qwen_r8_job.py'), '--system', 'ours', '--ours-mode', 'A',
               '--job-label', label, '--attempt', str(rep), '--ours-policy', policy, '--repeats', '1',
               '--quiet-2367', '--inline-demand-h2d', '--h2d-serial-ablation'] + hog
        cmd += ['--smoke'] if smoke else ['--post-generation-diagnostic']
        env = dict(os.environ)
    else:
        cmd = [PY, '-u', str(PKG / 'scripts/run_deepseek_r8_job.py'), '--decode-policy', policy, '--repeats', '1',
               '--attempt', str(rep), '--inline-demand-h2d', '--h2d-serial-ablation', '--job-label', label] + hog
        cmd += ['--smoke'] if smoke else ['--teacher-tokens', str(DS_TEACHER)]
        env = dict(os.environ, MGO_DEEPSEEK_CACHE_DIAG='1')
    env['MGO_RANDOM_SALT'] = str(rep)  # random arms draw differently per rep
    if table is not None:
        cmd += ['--quota-table', str(table)]
    subprocess.run(cmd, check=True, cwd=PKG.parent, env=env)
    return out


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status)
    try:
        job('qwen', 'l17', 'split', 1, smoke=True)  # Qwen guard + hog + SPLIT path
        for rep, order in enumerate(ORDERS, 1):
            for level in ('l13', 'l17'):
                for model in ('qwen', 'deepseek'):
                    for arm in order:
                        assert not (HERE / 'STOP').exists()
                        out = job(model, level, arm, rep)
                        status['jobs'].append(dict(model=model, level=level, arm=arm, policy=ARMS[arm], rep=rep, output=str(out)))
                        write(status); print('PASS', model, level, arm, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
