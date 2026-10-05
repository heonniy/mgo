"""Optional fixed placement of a two-thread CPU copy team, before inference."""
import os,threading
from pathlib import Path
import torch

def initialize(stage,cpus):
 cpus=tuple(cpus)
 if len(cpus)!=2 or len(set(cpus))!=2 or torch.get_num_threads()!=2:
  raise ValueError('fixed staging team requires two distinct CPUs and two Torch threads')
 tid=threading.get_native_id();os.sched_setaffinity(tid,cpus)
 before={int(p.name) for p in Path('/proc/self/task').iterdir()}
 # Materialize the same copy pool outside measured generation. A large CPU
 # zero fills only a private empty stage; it never touches model weights.
 stage.zero_()
 after={int(p.name) for p in Path('/proc/self/task').iterdir()}
 helpers=after-before
 if len(helpers)!=1:raise RuntimeError(f'expected one new copy helper, found {len(helpers)}; cannot establish fixed team safely')
 helper=helpers.pop();os.sched_setaffinity(helper,[cpus[1]]);os.sched_setaffinity(tid,[cpus[0]])
 return dict(staging_tid=tid,helper_tid=helper,staging_cpu=cpus[0],helper_cpu=cpus[1],stage_mask=sorted(os.sched_getaffinity(tid)),helper_mask=sorted(os.sched_getaffinity(helper)))
