"""One bounded, serial pass through the remaining authorized headline cells.

Failed jobs are preserved; dependent jobs are skipped for agent repair. This
driver never repeats an existing label and never runs two GPU jobs together.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/home/hwlee/mgo-results/headline_r4_20261007')
SCRIPT = Path(__file__).resolve().with_name('run_headline_job.py')
PYTHON = '/home/hwlee/sub-moe/phase01/.venv/bin/python'
TOOLS = Path('/home/hwlee/mgo-tools/headline-r4')
SMALL = 'R4_C30_B16_L256_O64'
LARGE = 'R4_C30_B64_L512_O64'


def jobs(after):
    return [
        ('infinity_B64_L512_primary1', 'MoE-Infinity-repaired', 'infinity', LARGE, 'infinity-env', 1, False, after),
        ('deepspeed_B16_L256_primary1', 'DeepSpeed-ZeRO-Inference', 'deepspeed', SMALL, 'base-env', 4, False, None),
        ('deepspeed_B64_L512_primary1', 'DeepSpeed-ZeRO-Inference', 'deepspeed', LARGE, 'base-env', 4, False, 'deepspeed_B16_L256_primary1'),
        ('llama_smoke1', 'llama.cpp-layer', 'llama', SMALL, 'base-env', 1, True, None),
        ('llama_B16_L256_primary1', 'llama.cpp-layer', 'llama', SMALL, 'base-env', 1, False, 'llama_smoke1'),
        ('llama_B64_L512_primary1', 'llama.cpp-layer', 'llama', LARGE, 'base-env', 1, False, 'llama_B16_L256_primary1'),
        ('ours_B64_L512_primary1', 'Ours', 'ours', LARGE, None, 4, False, None),
    ]


def state(label):
    path = ROOT / label / 'status.json'
    return json.loads(path.read_text()) if path.exists() else None


def main(a):
    queue = jobs(a.after)
    if a.dry_run:
        print(json.dumps(queue, indent=2))
        return
    assert not a.output.exists(), 'preserve the previous queue receipt'
    receipt = dict(status='WAITING', after=a.after, jobs=[], started=time.time())

    def save():
        temp = a.output.with_suffix('.tmp')
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(a.output)

    save()
    while True:
        if (ROOT / 'STOP').exists():
            receipt['status'] = 'STOPPED'
            save()
            return
        prior = state(a.after)
        assert prior is not None, 'the predecessor must already exist'
        if prior['status'] in ('PASS', 'FAIL'):
            break  # supervisor writes terminal status after restoring idle jobs
        time.sleep(5)
    for label, system, worker, cell, env, ranks, smoke, dependency in queue:
        if (ROOT / 'STOP').exists():
            receipt['status'] = 'STOPPED'
            save()
            return
        if dependency and (state(dependency) or {}).get('status') != 'PASS':
            receipt['jobs'].append(dict(label=label, status='SKIPPED_DEPENDENCY', dependency=dependency))
            save()
            continue
        if (ROOT / label).exists():
            assert (state(label) or {}).get('status') in ('PASS', 'FAIL'), 'existing job is not terminal'
            receipt['jobs'].append(dict(label=label, status='EXISTING', result=state(label)))
            save()
            continue
        command = [PYTHON, str(SCRIPT), '--label', label, '--system', system,
                   '--worker', f'headline_{worker}_worker.py', '--cell', cell,
                   '--python', str(TOOLS / env / 'bin/python') if env else PYTHON,
                   '--ranks', str(ranks), '--timeout', '7200']
        if smoke:
            command.append('--smoke')
        receipt.update(status='RUNNING', current=label)
        save()
        result = subprocess.run(command, check=False)
        receipt['jobs'].append(dict(label=label, returncode=result.returncode, result=state(label)))
        save()
    receipt.update(status='FINISHED', finished=time.time(), current=None)
    save()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--after', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    main(p.parse_args())
