"""Exercise the late-created native helper inheritance that broke V2."""
import os,threading
from refactor_thread_affinity import configure

def main():
 original=sorted(os.sched_getaffinity(0));assert len(original)>=4
 cpus=original[:4];ready=threading.Event();launch=threading.Event();done=threading.Event();stop=threading.Event();observed=[]
 def stage():
  ready.set();launch.wait()
  helper=threading.Thread(target=lambda:observed.append(sorted(os.sched_getaffinity(0))))
  helper.start();helper.join();done.set();stop.wait()
 worker=threading.Thread(target=stage);worker.start();ready.wait()
 try:
  row=configure(cpus,worker.native_id,True)
  assert row['main']==cpus[:1] and row['staging']==cpus[1:3]
  launch.set();assert done.wait(5)
  assert observed==[cpus[1:3]],observed
  assert not set(row['main'])&set(row['staging'])
  row=configure(cpus,worker.native_id,False)
  assert row['main']==row['staging']==cpus
 finally:
  launch.set();stop.set();worker.join();os.sched_setaffinity(0,original)
 print('PASS: late helper inherits two staging CPUs; main disjoint; restoration complete')
if __name__=='__main__':main()
