"""Add unconstrained-quota arms to the stage breakdown (same environment, 3 interleaved reps, both models).

- rquota = RANDOM_QUOTA_NEAR: i.i.d. uniform rank per miss -> unbalanced quota, Near placement inside it.
(A RANDOM / policy-19 arm was dropped at the owner's request before any RANDOM job started.)
Reuses run.py's job launchers. STATUS_extra.json is separate so the 30-job STATUS.json stays unchanged.
"""
import json, time
from pathlib import Path

import run as base

HERE = Path(__file__).resolve().parent
base.ARMS.update({'rquota': 'RANDOM_QUOTA_NEAR'})
ORDERS = (('rquota',), ('rquota',), ('rquota',))


def write(data):
    p = HERE / 'STATUS_extra.json'; t = p.with_suffix('.json.tmp'); t.write_text(json.dumps(data, indent=2) + '\n'); t.replace(p)


def main():
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status)
    try:
        for rep, order in enumerate(ORDERS, 1):
            for model, fn in (('qwen', base.qwen), ('deepseek', base.deepseek)):
                for tag in order:
                    assert not (HERE / 'STOP').exists()
                    out = fn(tag, rep)
                    status['jobs'].append(dict(model=model, arm=tag, policy=base.ARMS[tag], rep=rep, output=str(out)))
                    write(status)
                    print('PASS', model, tag, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status)


if __name__ == '__main__':
    main()
