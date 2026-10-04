"""Wait for CPU selection, then build plans without starting GPU measurements."""
from prepare_old_ca_fanout import ROOT,PACKET,P,PYTHON,json,write
import subprocess,time
from pathlib import Path
import psutil

def main():
 meta=json.loads((ROOT/'process.json').read_text());deadline=time.monotonic()+7200
 while not (ROOT/'status.json').exists():
  if (ROOT/'STOP').exists():raise RuntimeError('Owner STOP')
  stat=Path('/proc')/str(meta['pid'])/'stat'
  if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0]=='Z':raise RuntimeError('CPU preparation exited without completion; inspect prepare.log')
  if time.monotonic()>deadline:raise TimeoutError('CPU preparation deadline')
  time.sleep(5)
 assert json.loads((ROOT/'status.json').read_text())['status']=='CPU_SELECTION_COMPLETE'
 assert psutil.virtual_memory().available>=256*2**30
 write(ROOT/'plan_build_status.json',dict(status='RUNNING',stage='THREE_POLICY_PLAN_BUILD'))
 subprocess.run([PYTHON,str(P/'scripts/build_old_ca_fanout_plans.py')],check=True)
 write(ROOT/'plan_build_status.json',dict(status='PLANS_READY',physical_started=False,finished_unix=time.time()))

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  write(ROOT/'plan_build_status.json',dict(status='FAILED_OR_STOPPED',error=repr(exc)));raise
