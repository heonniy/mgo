"""Actual CUDA interval unions from Nsight SQLite; no CPU-span-as-kernel proxy."""
import argparse,bisect,json,sqlite3
from collections import defaultdict
from pathlib import Path

def union(intervals):
 out=[]
 for a,b in sorted(intervals):
  if b<=a:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return out

def duration(intervals):return sum(b-a for a,b in union(intervals))
def intersection(left,right):
 a,b=union(left),union(right);i=j=0;out=[]
 while i<len(a) and j<len(b):
  lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
  if hi>lo:out.append((lo,hi))
  if a[i][1]<=b[j][1]:i+=1
  else:j+=1
 return out

def analyze(path):
 db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
 ranges=defaultdict(list);decode=[];cpu=defaultdict(list)
 for r in db.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL AND end>start'):
  name=strings.get(r['textId'],r['text'])
  if not name:continue
  if name.startswith('decode.event.'):
   decode.append((r['start'],r['end']));name='decode.event'
  ranges[(r['globalTid'],name)].append((r['start'],r['end']))
  if name.startswith('moe.'):cpu[name].append((r['start'],r['end']))
 assert len(decode)==384,('expected eight decode steps, 48 layers',len(decode))
 window=[(min(a for a,b in decode),max(b for a,b in decode))]
 for key,value in list(ranges.items()):
  value.sort();ranges[key]=([a for a,b in value],value)
 def contained(tid,name,start,end):
  pair=ranges.get((tid,name))
  if pair is None:return False
  starts,intervals=pair;i=bisect.bisect_right(starts,start)-1
  return i>=0 and intervals[i][1]>=end
 runtime={}
 for r in db.execute('SELECT start,end,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
  key=(r['globalTid'] & 0xFFFFFFFFFF000000,r['correlationId'])
  runtime[key]=(r['start'],r['end'],r['globalTid'])
 phases=('moe.metadata','moe.forward_a2a','moe.expert_compute','moe.return_a2a')
 kernels=defaultdict(list);counts=defaultdict(int);unassigned=0;decode_kernels=[]
 for r in db.execute('SELECT start,end,globalPid,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
  api=runtime.get((r['globalPid'],r['correlationId']))
  if api is None:continue
  start,end,tid=api
  if not contained(tid,'decode.event',start,end):continue
  interval=(r['start'],r['end']);decode_kernels.append(interval);matched=False
  for name in phases:
   if contained(tid,name,start,end):kernels[name].append(interval);counts[name]+=1;matched=True
  if not matched:unassigned+=1
 assert counts['moe.expert_compute']>0,'missing expert kernel attribution'
 assert counts['moe.forward_a2a']>0 and counts['moe.return_a2a']>0,'missing collective attribution'
 # CUDA memcpy enum names are provided by Nsight, not guessed from kernel names.
 kinds={r['id']:r['label'] for r in db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER')}
 h2d=[];h2d_bytes=0;h2d_count=0
 for r in db.execute('SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
  label=kinds[r['copyKind']].lower().replace(' ','')
  if label not in ('htod','h2d','hosttodevice'):continue
  clipped=intersection([(r['start'],r['end'])],window)
  if clipped:h2d+=clipped;h2d_count+=1;h2d_bytes+=r['bytes']
 assert h2d,'missing actual H2D DMA events'
 comm=kernels['moe.forward_a2a']+kernels['moe.return_a2a'];expert=kernels['moe.expert_compute'];cover=comm+expert
 total=duration(h2d);hidden=duration(intersection(h2d,cover));cpu_report={}
 for name,ints in cpu.items():
  ints=intersection(ints,window);full=duration(ints);overlap=duration(intersection(ints,cover));cpu_report[name]=dict(total_ms=full/1e6,overlap_comm_expert_ms=overlap/1e6,outside_comm_expert_ms=(full-overlap)/1e6)
 return dict(status='PASS',source=str(path),decode_events=len(decode),window_ns=window,kernel_counts=dict(counts),unassigned_decode_kernel_count=unassigned,kernel_union_ms={k:duration(v)/1e6 for k,v in kernels.items()},all_decode_kernel_union_ms=duration(decode_kernels)/1e6,H2D=dict(count=h2d_count,bytes_for_events_intersecting_window=h2d_bytes,total_union_ms=total/1e6,overlap_comm_expert_ms=hidden/1e6,outside_comm_expert_ms=(total-hidden)/1e6,hidden_ratio=hidden/total),cpu_nvtx=cpu_report,interpretation='Overlap is interval intersection, not a causal speedup or critical-path proof. NCCL kernel residency includes wait/spin time. Copy intervals are clipped to the decode CPU-event window. GPU phases use actual correlated CUDA kernels; CPU NVTX spans are reported separately. Never use these instrumented times as primary E2E/TPOT.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(analyze(a.sqlite),indent=2)+'\n')
