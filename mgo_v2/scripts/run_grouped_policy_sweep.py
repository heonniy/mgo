"""Serial guarded Qwen R4 B8/B16/B64 placement and grouped-executor study."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path('/home/hwlee/mgo-results/grouped_policy_scaling_20261009')
WORKLOADS = ROOT/'WORKLOADS.json'
POLICIES = ('BR', 'CA_NATIVE', 'LA_CA_NEAR')
CELLS = {batch: f'Qwen3_ShareGPT_R4_C30_B{batch}_L512_O64'
         for batch in (8, 16, 64)}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def run(job, arm, batch, policy, diagnostic):
    kind = 'diagnostic' if diagnostic else 'full'
    name = f'r4_{arm.lower()}_{policy.lower()}_{batch}_{kind}_v1'
    path = ROOT/'jobs'/name
    if path.exists():
        state = json.loads((path/'status.json').read_text())
        assert state['status'] == 'PASS', f'existing job is not reusable: {path}'
        return path
    command = [sys.executable, str(HERE/'run_ours_grouping_r4.py'),
               '--arm', arm, '--cell', CELLS[batch], '--policy', policy,
               '--workloads', str(WORKLOADS), '--output-root', str(ROOT),
               '--attempt', '1', '--repeats', '1' if diagnostic else '2']
    if arm == 'N':
        command.append('--compiled-dense')
    if diagnostic:
        command.append('--diagnostic')
    write(ROOT/'queue_status.json', dict(status='RUNNING', job=job, command=command,
                                         started=time.time()))
    subprocess.run(command, check=True, env=os.environ.copy())
    assert json.loads((path/'status.json').read_text())['status'] == 'PASS'
    return path


def main():
    assert json.loads(WORKLOADS.read_text())['status'] == 'FROZEN'
    queue = [(8,'A','LA_CA_NEAR')]
    queue += [(batch,'N',policy) for batch in (8,) for policy in POLICIES]
    queue += [(16,'A','LA_CA_NEAR')]
    queue += [(batch,'N',policy) for batch in (16,64) for policy in POLICIES]
    done = []
    for batch, arm, policy in queue:
        job = dict(batch=batch, arm=arm, policy=policy)
        primary = run(job,arm,batch,policy,False)
        diagnostic = run(job,arm,batch,policy,True)
        report = ROOT/'reports'/f'{arm.lower()}_{policy.lower()}_b{batch}.json'
        subprocess.run([sys.executable, str(HERE/'summarize_r4_ep_diagnostic.py'),
                        '--primary',str(primary),'--diagnostic',str(diagnostic),
                        '--output',str(report)],check=True)
        done.append(str(report))
        write(ROOT/'queue_status.json', dict(status='RUNNING', completed=done,
                                             next_index=len(done), total=len(queue)))
    write(ROOT/'queue_status.json', dict(status='PASS', completed=done,
                                         next_index=len(done), total=len(queue)))


if __name__ == '__main__':
    main()
