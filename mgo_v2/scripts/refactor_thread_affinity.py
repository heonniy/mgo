"""Diagnostic process-local CPU placement; never changes host policy."""
import os,threading
from pathlib import Path

def configure(cpus,staging_tid,isolated,fixed_team=None):
 cpus=sorted(set(cpus));assert len(cpus)>=4
 main_tid=threading.get_native_id();assert staging_tid is not None and staging_tid!=main_tid
 helper=cpus[3:] if isolated else cpus
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),helper)
  except ProcessLookupError:pass
 os.sched_setaffinity(main_tid,[cpus[0]] if isolated else cpus)
 os.sched_setaffinity(staging_tid,cpus[1:3] if isolated else cpus)
 if fixed_team is not None:
  assert isolated and fixed_team['staging_tid']==staging_tid
  assert [fixed_team['staging_cpu'],fixed_team['helper_cpu']]==cpus[1:3]
  os.sched_setaffinity(staging_tid,[cpus[1]])
  os.sched_setaffinity(fixed_team['helper_tid'],[cpus[2]])
 return dict(fixed_team=fixed_team,isolated=isolated,main=sorted(os.sched_getaffinity(main_tid)),staging=sorted(os.sched_getaffinity(staging_tid)),helpers=helper)
