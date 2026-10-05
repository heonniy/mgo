"""Diagnostic process-local CPU placement; never changes host policy."""
import os,threading
from pathlib import Path

def configure(cpus,staging_tid,isolated):
 cpus=sorted(set(cpus));assert len(cpus)>=3
 main_tid=threading.get_native_id();assert staging_tid is not None and staging_tid!=main_tid
 helper=cpus[2:] if isolated else cpus
 for task in Path('/proc/self/task').iterdir():
  try:os.sched_setaffinity(int(task.name),helper)
  except ProcessLookupError:pass
 os.sched_setaffinity(main_tid,[cpus[0]] if isolated else cpus)
 os.sched_setaffinity(staging_tid,[cpus[1]] if isolated else cpus)
 return dict(isolated=isolated,main=sorted(os.sched_getaffinity(main_tid)),staging=sorted(os.sched_getaffinity(staging_tid)),helpers=helper)
