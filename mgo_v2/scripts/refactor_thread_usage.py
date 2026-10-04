"""Two own-thread snapshots outside timed generation; no continuous sampling."""
import os,threading
from pathlib import Path
HZ=os.sysconf('SC_CLK_TCK')
def snapshot(staging_tid):
 try:enabled=Path('/proc/sys/kernel/sched_schedstats').read_text().strip()=='1'
 except OSError:enabled=None
 rows={}
 for role,tid in [('main',threading.get_native_id()),('expert_staging',staging_tid)]:
  if tid is None:continue
  path=Path('/proc/self/task')/str(tid)
  try:
   fields=(path/'stat').read_text().rsplit(')',1)[1].split()
   runtime,waiting,slices=map(int,(path/'schedstat').read_text().split())
  except OSError:continue
  rows[role]=dict(tid=tid,start_ticks=int(fields[19]),user_seconds=int(fields[11])/HZ,system_seconds=int(fields[12])/HZ,scheduler_runtime_ns=runtime,runqueue_wait_ns=waiting if enabled else None,timeslices=slices,last_cpu=int(fields[36]))
 return dict(schedstats_enabled=enabled,threads=rows)

def delta(before,after):
 out={}
 for role,b in before['threads'].items():
  a=after['threads'].get(role)
  if a is None or (a['tid'],a['start_ticks'])!=(b['tid'],b['start_ticks']):continue
  row={key:a[key]-b[key] for key in ('user_seconds','system_seconds','scheduler_runtime_ns','timeslices')}
  row.update(runqueue_wait_ns=None if a['runqueue_wait_ns'] is None or b['runqueue_wait_ns'] is None else a['runqueue_wait_ns']-b['runqueue_wait_ns'],cpu_before=b['last_cpu'],cpu_after=a['last_cpu'])
  out[role]=row
 return dict(threads=out,schedstats_enabled=before['schedstats_enabled'] and after['schedstats_enabled'],scope='Snapshots bracket generation outside its E2E/TPOT timers. Main and expert-staging threads only; includes untimed generation entry/exit work. Null runqueue wait means kernel scheduler statistics are disabled, not zero wait.')
