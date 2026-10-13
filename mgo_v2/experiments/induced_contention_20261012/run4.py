"""No-hog Qwen balanced random with ordinary salts 2 and 3 (salt = rep); rep 1 (salt 1) is campaign 2's icql00brand_full_v1."""
import time
from run import job, write

status = {'status': 'RUNNING', 'started': time.time(), 'jobs': []}
write(status, 'STATUS4.json')
try:
    for rep in (2, 3):
        out = job('qwen', 'l00', 'brand', rep)
        status['jobs'].append(dict(model='qwen', level='l00', arm='brand', rep=rep, salt=rep, output=str(out)))
        write(status, 'STATUS4.json'); print('PASS l00 brand', rep, flush=True)
    status['status'] = 'PASS'
except BaseException as e:
    status.update(status='FAIL', error=repr(e)); raise
finally:
    status['finished'] = time.time(); write(status, 'STATUS4.json')
