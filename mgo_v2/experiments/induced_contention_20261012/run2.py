"""Second campaign (after run.py): no-hog NEAR/FAST/SPLIT(1.44x table)/balanced-random, and balanced-random
at the 1.3x and 1.7x hog levels. 2 models x 3 reps, 36 jobs, arm order rotated per rep."""
import json, time
from run import HERE, job, write

PLAN = [('l00', ('near', 'fast', 'split', 'brand')), ('l13', ('brand',)), ('l17', ('brand',))]


def main():
    while json.loads((HERE / 'STATUS.json').read_text())['status'] == 'RUNNING':
        time.sleep(30)  # GPUs are exclusive: wait for the first campaign
    status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
    write(status, 'STATUS2.json')
    try:
        for rep in (1, 2, 3):
            for level, arms in PLAN:
                arms = arms[rep - 1:] + arms[:rep - 1]
                for model in ('qwen', 'deepseek'):
                    for arm in arms:
                        assert not (HERE / 'STOP').exists()
                        out = job(model, level, arm, rep)
                        status['jobs'].append(dict(model=model, level=level, arm=arm, rep=rep, output=str(out)))
                        write(status, 'STATUS2.json'); print('PASS', model, level, arm, rep, flush=True)
        status['status'] = 'PASS'
    except BaseException as e:
        status.update(status='FAIL', error=repr(e)); raise
    finally:
        status['finished'] = time.time(); write(status, 'STATUS2.json')


if __name__ == '__main__':
    main()
