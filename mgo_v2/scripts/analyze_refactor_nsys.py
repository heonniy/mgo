"""Actual CUDA interval unions from Nsight SQLite; no CPU-span-as-kernel proxy."""
import argparse,bisect,hashlib,json,sqlite3,statistics
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

class IntervalIndex:
 """Clip disjoint unions without rescanning an entire long capture per layer."""
 def __init__(self,intervals):
  self.intervals=union(intervals)
  self.starts=[a for a,b in self.intervals];self.ends=[b for a,b in self.intervals]
 def clip(self,start,end):
  lo=bisect.bisect_right(self.ends,start);hi=bisect.bisect_left(self.starts,end)
  return [(max(start,a),min(end,b)) for a,b in self.intervals[lo:hi]]

def exclusive_partition(window,h2d,comm,compute):
 """Disjoint actual-time bins, including activity outside the three classes."""
 names=('idle_or_unattributed','H2D_only','COMM_only','H2D_COMM',
        'COMPUTE_only','H2D_COMPUTE','COMM_COMPUTE','H2D_COMM_COMPUTE')
 totals={name:0 for name in names}
 # Union each class first: concurrent kernels of one class count only once.
 events=defaultdict(lambda:[0,0,0])
 for k,intervals in enumerate((h2d,comm,compute)):
  for a,b in intersection(intervals,window):
   events[a][k]+=1;events[b][k]-=1
 for a,b in union(window):
  events[a];events[b]
 active=[0,0,0];previous=None
 for t,delta in sorted(events.items()):
  if previous is not None:
   inside=duration(intersection([(previous,t)],window))
   mask=sum(1<<k for k,v in enumerate(active) if v)
   totals[names[mask]]+=inside
  active=[v+d for v,d in zip(active,delta)];previous=t
 assert sum(totals.values())==duration(window)
 return {name:value/1e6 for name,value in totals.items()}

def analyze(path,receipt):
 expected_events=48*receipt['case']['horizon']
 db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 strings={r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
 ranges=defaultdict(list);decode=[];decode_ids=[];cpu=defaultdict(list);event_ids={}
 for r in db.execute('SELECT start,end,globalTid,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL AND end>start'):
  name=strings.get(r['textId'],r['text'])
  if not name:continue
  if name.startswith('decode.event.'):
   event_ids[(r['globalTid'],r['start'])]=int(name.rsplit('.',1)[1])
   decode_ids.append((int(name.rsplit('.',1)[1]),r['start'],r['end']))
   decode.append((r['start'],r['end']));name='decode.event'
  ranges[(r['globalTid'],name)].append((r['start'],r['end']))
  if name.startswith('moe.'):cpu[name].append((r['start'],r['end']))
 assert len(decode)==expected_events,('wrong captured horizon',len(decode),expected_events)
 window=[(min(a for a,b in decode),max(b for a,b in decode))]
 for key,value in list(ranges.items()):
  value.sort();ranges[key]=([a for a,b in value],value)
 def contained(tid,name,start,end):
  pair=ranges.get((tid,name))
  if pair is None:return False
  starts,intervals=pair;i=bisect.bisect_right(starts,start)-1
  return i>=0 and intervals[i][1]>=end
 runtime={};tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
 for api_table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
  if api_table not in tables:continue
  for r in db.execute(f'SELECT start,end,globalTid,correlationId FROM {api_table}'):
   key=(r['globalTid'] & 0xFFFFFFFFFF000000,r['correlationId'])
   runtime.setdefault(key,(r['start'],r['end'],r['globalTid']))
 phases=('moe.metadata','moe.forward_a2a','moe.expert_compute','moe.post_expert_global_barrier','moe.return_a2a')
 kernels=defaultdict(list);counts=defaultdict(int);kernel_names=defaultdict(lambda:defaultdict(int));unassigned=0;decode_kernels=[];event_kernels=defaultdict(lambda:defaultdict(list))
 for r in db.execute('SELECT start,end,globalPid,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL'):
  api=runtime.get((r['globalPid'],r['correlationId']))
  if api is None:continue
  start,end,tid=api
  if not contained(tid,'decode.event',start,end):continue
  event_starts,event_spans=ranges[(tid,'decode.event')]
  event=event_ids[(tid,event_starts[bisect.bisect_right(event_starts,start)-1])]
  interval=(r['start'],r['end']);decode_kernels.append(interval);matched=False
  for name in phases:
   if contained(tid,name,start,end):
    kernel_name=strings.get(r['demangledName'],'<unknown>');phase=name
    if name in ('moe.metadata','moe.forward_a2a','moe.post_expert_global_barrier','moe.return_a2a'):phase += '.nccl' if 'nccl' in kernel_name.lower() else '.local_gpu'
    kernels[phase].append(interval);counts[phase]+=1;kernel_names[phase][kernel_name]+=1;matched=True
    event_kernels[event][phase].append(interval)
  if not matched:unassigned+=1
 assert counts['moe.expert_compute']>0,'missing expert kernel attribution'
 assert counts['moe.metadata.nccl']>0,'missing metadata collective attribution'
 assert counts['moe.forward_a2a.nccl']>0 and counts['moe.return_a2a.nccl']>0,'missing collective attribution'
 # CUDA memcpy enum names are provided by Nsight, not guessed from kernel names.
 kinds={r['id']:r['label'] for r in db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER')}
 h2d=[];h2d_bytes=0;h2d_count=0;other_h2d=[];other_bytes=0;other_count=0
 for r in db.execute('SELECT start,end,copyKind,bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start,end'):
  label=kinds[r['copyKind']].lower().replace(' ','').replace('-','')
  if label not in ('htod','h2d','hosttodevice'):continue
  if r['end']<=window[0][0]:continue
  assert r['start']>=window[0][0],'H2D crosses synchronized prefill/decode boundary'
  interval=(r['start'],r['end'])
  if r['bytes']==9437184:h2d.append(interval);h2d_count+=1;h2d_bytes+=r['bytes']
  else:other_h2d.append(interval);other_count+=1;other_bytes+=r['bytes']
 assert h2d,'missing actual expert-weight H2D DMA events'
 assert receipt['status']=='PASS'
 assert h2d_count==receipt['decode_expert_copies'],('CUDA expert-copy count mismatch',h2d_count,receipt['decode_expert_copies'])
 assert h2d_bytes==receipt['decode_expert_bytes']
 trace_path=Path(receipt['copy_trace_path']);trace_raw=trace_path.read_bytes()
 assert hashlib.sha256(trace_raw).hexdigest()==receipt['copy_trace_sha256']
 trace=json.loads(trace_raw);assert len(trace)==len(h2d)
 split=defaultdict(list);byevent=defaultdict(list);readiness=defaultdict(int);readiness_intervals=defaultdict(list)
 for interval,row in zip(h2d,trace):
  assert row['bytes']==9437184
  split[row['kind']].append(interval);byevent[row['source_event']].append(interval)
  if row['readiness_at_use'] is not None:
   readiness[row['readiness_at_use']]+=1;readiness_intervals[row['readiness_at_use']].append(interval)

 # The capture ends only after generation and scheduler synchronization. Keep
 # trailing decode DMA/kernel work instead of clipping to CPU launch spans.
 window=[(window[0][0],max([window[0][1]]+[b for a,b in h2d+other_h2d+decode_kernels]))]
 comm=kernels['moe.forward_a2a.nccl']+kernels['moe.return_a2a.nccl'];expert=kernels['moe.expert_compute'];cover=comm+expert
 total=duration(h2d);hidden=duration(intersection(h2d,cover));cpu_report={}
 decomposition_comm=comm+kernels['moe.metadata.nccl']
 exclusive=exclusive_partition(window,h2d,decomposition_comm,expert)
 # Adjacent host event-start boundaries partition the complete decode window;
 # GPU work is assigned by actual timestamp, not the launch's layer label.
 # A copy spanning a boundary is split, never counted twice.
 ordered=sorted(decode_ids,key=lambda x:x[1]);per_event=[]
 indexed=[IntervalIndex(intervals) for intervals in (h2d,decomposition_comm,expert)]
 for i,(event,start,end) in enumerate(ordered):
  stop=ordered[i+1][1] if i+1<len(ordered) else window[0][1]
  per_event.append(dict(event=event,layer=event%48,step=event//48-1,
                        window_ns=[start,stop],exclusive_ms=exclusive_partition([(start,stop)],*[index.clip(start,stop) for index in indexed])))
 for key,value in exclusive.items():
  assert abs(sum(row['exclusive_ms'][key] for row in per_event)-value)<1e-6
 split_report={kind:dict(count=len(ints),bytes=len(ints)*9437184,total_union_ms=duration(ints)/1e6,overlap_comm_expert_ms=duration(intersection(ints,cover))/1e6) for kind,ints in split.items()}
 split_report['readiness_at_use']={kind:dict(count=len(ints),bytes=len(ints)*9437184,total_union_ms=duration(ints)/1e6,overlap_comm_expert_ms=duration(intersection(ints,cover))/1e6) for kind,ints in readiness_intervals.items()}
 split_report['readiness_scope']='Observed at logical promotion in the current-layer controller, before forward payload launch. Classified DMA duration covers the whole copy, not remaining wait time after promotion or expert-kernel start. Queued copies canceled before submission are absent from DMA classes.'
 # One all-gather per decode metadata record. Distinguish host packet
 # preparation, the collective API call, and blocking readback plus CPU unpack.
 # The final category is intentionally not presented as pure communication.
 if 'moe.metadata_collective_submit' in cpu:
  partitioned=0
  for (tid,name),(starts,parents) in list(ranges.items()):
   if name!='moe.metadata':continue
   child_pair=ranges.get((tid,'moe.metadata_collective_submit'))
   assert child_pair is not None
   child_starts,children=child_pair
   for a,b in parents:
    if not contained(tid,'decode.event',a,b):continue
    i=bisect.bisect_left(child_starts,a);j=bisect.bisect_left(child_starts,b)
    assert j-i==1 and children[i][1]<=b,'expected one nested decode metadata collective'
    c,d=children[i];cpu['moe.metadata_pack_host'].append((a,c))
    cpu['moe.metadata_readback_wait_and_unpack'].append((d,b));partitioned+=1
  assert partitioned==expected_events
 for name,ints in cpu.items():
  ints=intersection(ints,window);full=duration(ints);overlap=duration(intersection(ints,cover));a2a_overlap=duration(intersection(ints,comm));cpu_report[name]=dict(total_ms=full/1e6,overlap_comm_expert_ms=overlap/1e6,outside_comm_expert_ms=(full-overlap)/1e6,overlap_payload_A2A_ms=a2a_overlap/1e6,outside_payload_A2A_ms=(full-a2a_overlap)/1e6)
 if 'moe.prefetch_predictor' in cpu_report:
  parent=intersection(cpu['moe.prefetch_controller'],window)
  child=intersection(cpu['moe.prefetch_predictor'],window)
  assert duration(intersection(parent,child))==duration(child),'predictor must be nested in prefetch controller'
  cpu_report['moe.prefetch_placement_and_bookkeeping']={key:max(0,value-cpu_report['moe.prefetch_predictor'][key]) for key,value in cpu_report['moe.prefetch_controller'].items()}
 decomposition=dict(aggregate_exclusive_ms=exclusive,per_event=per_event,window_definition='Adjacent host decode-event start timestamps, through final captured decode work. Actual GPU intervals are split at boundaries; these are time windows, not causal layer ownership. COMM includes metadata and both payload NCCL collectives. Local pack/combine kernels and non-MoE work remain unattributed. Idle_or_unattributed is not necessarily GPU idle.')
 causal={str(event):{phase:duration(intervals)/1e6 for phase,intervals in phases.items()} for event,phases in event_kernels.items()}
 assert len(causal)==expected_events
 distributions={}
 for phase in kernels:
  values=[phases.get(phase,0.) for phases in causal.values()]
  distributions[phase]=dict(median_ms=statistics.median(values),p90_ms=statistics.quantiles(values,n=10,method='inclusive')[8],max_ms=max(values),events=len(values))
 decomposition['causal_event_kernel_union_ms']=causal
 decomposition['causal_event_kernel_distributions']=distributions
 decomposition['causal_scope']='Actual GPU intervals attributed to their CPU launch event. Unlike adjacent-time-window bins, causal event durations can overlap and must not be summed into decode wall time. NCCL durations include residency/wait time.'
 preview_window=[(ordered[0][1],ordered[3][1])]
 preview_classes={**kernels, 'expert_H2D_DMA':h2d}
 for phase in ('moe.current_controller','moe.prefetch_controller','moe.metadata'):
  preview_classes['CPU '+phase]=cpu[phase]
 decomposition['timeline_preview']=dict(window_ns=preview_window[0],
  intervals_ns={phase:intersection(intervals,preview_window) for phase,intervals in preview_classes.items()},
  event_boundaries_ns=[dict(event=e,start_ns=s) for e,s,end in ordered[:4]],
  scope='First three adjacent decode-layer time windows. GPU lanes use actual CUDA intervals; CPU lanes use NVTX wall intervals. Not primary timing or a critical-path proof.')
 return dict(status='PASS',interval_decomposition=decomposition,source=str(path),decode_events=len(decode),window_ns=window,kernel_counts=dict(counts),kernel_names={k:dict(v) for k,v in kernel_names.items()},prefetch_readiness_at_use=dict(readiness),H2D_split=split_report,H2D_by_source_event_ms={str(k):duration(v)/1e6 for k,v in byevent.items()},unassigned_decode_kernel_count=unassigned,kernel_union_ms={k:duration(v)/1e6 for k,v in kernels.items()},all_decode_kernel_union_ms=duration(decode_kernels)/1e6,H2D=dict(count=h2d_count,bytes=h2d_bytes,total_union_ms=total/1e6,overlap_comm_expert_ms=hidden/1e6,outside_comm_expert_ms=(total-hidden)/1e6,hidden_ratio=hidden/total),other_H2D=dict(count=other_count,bytes=other_bytes,total_union_ms=duration(other_h2d)/1e6),cpu_nvtx=cpu_report,interpretation='Overlap is interval intersection, not a causal speedup or critical-path proof. NCCL kernel residency includes wait/spin time. Expert DMA uses the 9-MiB copy size and exactly reconciles the scheduler decode-copy count; smaller control/input copies are reported separately. Capture includes trailing decode work after CPU launch spans. NCCL kernel names separate collective work from local pack/combine kernels. Copy classes map the single H2D stream in submission order to checked ticket provenance. GPU phases use actual correlated CUDA runtime/driver launches; CPU NVTX spans are reported separately. Never use these instrumented times as primary E2E/TPOT.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('sqlite',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();a.output.write_text(json.dumps(analyze(a.sqlite,json.loads(a.receipt.read_text())),indent=2)+'\n')
