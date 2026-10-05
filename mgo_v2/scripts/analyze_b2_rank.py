"""Join diagnostic host/NVTX/CUPTI evidence and align to shared host raw clock."""
import bisect,json,sqlite3
from collections import defaultdict
from pathlib import Path
import numpy as np
from analyze_refactor_nsys import union,duration,intersection

def analyze(capture,rank):
 host=json.loads((capture/f'b2_host_rank{rank}.json').read_text());byid={x['serial']:x for x in host['rows']}
 dbpath=capture/f'interval_analysis/rank{rank}.sqlite';db=sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 strings=dict(db.execute('select id,value from StringIds'));ranges=defaultdict(list);anchors=[]
 for r in db.execute('select start,end,globalTid,text,textId from NVTX_EVENTS where end>start'):
  name=strings.get(r['textId'],r['text']) or ''
  if not name.startswith('b2|'):continue
  _,event,serial,phase=name.split('|');serial=int(serial);x=byid[serial];assert x['event']==int(event) and x['phase']==phase
  ranges[r['globalTid']].append((r['start'],r['end'],serial))
  anchors.append(((r['start']+r['end'])/2,(x['host_start_raw_ns']+x['host_end_raw_ns'])/2))
  x['nvtx_start_ns']=r['start'];x['nvtx_end_ns']=r['end']
 assert len(anchors)==len(byid)
 ns,raw=np.array(anchors).T;origin=ns.mean();raworigin=raw.mean();slope=float(np.dot(ns-origin,raw-raworigin)/np.dot(ns-origin,ns-origin));offset=float(np.median(raw-slope*ns));error=raw-(slope*ns+offset)
 assert .999<slope<1.001,('clock drift',slope)
 clock=dict(slope=slope,offset_ns=offset,p50_anchor_residual_us=float(np.quantile(abs(error),.5)/1000),p99_anchor_residual_us=float(np.quantile(abs(error),.99)/1000),max_anchor_residual_us=float(max(abs(error))/1000),anchors=len(anchors),clock='CUPTI Nsight relative timestamp mapped to shared CLOCK_MONOTONIC_RAW by matched NVTX/host-span midpoints; anchor error retained.')
 assert clock['p99_anchor_residual_us']<=50,('clock uncertainty too large',clock)
 for tid,rs in list(ranges.items()):
  rs.sort();ranges[tid]=([x[0] for x in rs],rs)
 def find(tid,a,b):
  pair=ranges.get(tid)
  if not pair:return None
  starts,rs=pair;i=bisect.bisect_right(starts,a)-1;best=None
  # At most one expert-loop parent plus nested selection/wait per event.
  # Scan until the event boundary; no fixed-span guess needed.
  event=None
  while i>=0:
   s,e,k=rs[i];row=byid[k]
   if event is None:event=row['event']
   if row['event']!=event:break
   if e>=b:return k
   i-=1
  return None if best is None else best[1]
 runtime={};tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
 for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
  if table in tables:
   for r in db.execute(f'select start,end,globalTid,correlationId from {table}'):
    runtime.setdefault((r['globalTid'] & 0xFFFFFFFFFF000000,r['correlationId']),(r['start'],r['end'],r['globalTid']))
 stream_ops=defaultdict(list);serial_stream={}
 kernels=defaultdict(list);apis=defaultdict(list);names=defaultdict(lambda:defaultdict(int))
 for r in db.execute('select start,end,globalPid,correlationId,demangledName,streamId from CUPTI_ACTIVITY_KIND_KERNEL'):
  stream_ops[r['streamId']].append((r['start'],r['end']))
  api=runtime.get((r['globalPid'],r['correlationId']))
  if not api:continue
  a,b,tid=api;k=find(tid,a,b)
  if k is None:continue
  name=strings.get(r['demangledName'],'');phase=byid[k]['phase']
  # Collective ranges also allocate tensors; separate only true NCCL kernels.
  if phase.endswith('collective_host_call') and 'nccl' not in name.lower():phase+=':local_gpu'
  serial_stream[k]=r['streamId'];kernels[k].append((r['start'],r['end'],phase));apis[k].append((a,b));names[phase][name]+=1
 def mapped(t):return int(round(slope*t+offset))
 copy_trace=json.loads((capture/f'copy_trace_rank{rank}.json').read_text());copy_rows=[]
 copy_kinds={r['id']:r['label'] for r in db.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')}
 first_nvtx=min(x['nvtx_start_ns'] for x in byid.values())
 for r in db.execute('select start,end,copyKind,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMCPY order by start,end'):
  stream_ops[r['streamId']].append((r['start'],r['end']))
  label=copy_kinds[r['copyKind']].lower().replace(' ','').replace('-','')
  if r['bytes']==9437184 and r['start']>=first_nvtx and label in ('htod','h2d','hosttodevice'):copy_rows.append((r['start'],r['end']))
 assert len(copy_rows)==len(copy_trace),('decode9MiB copy mismatch',len(copy_rows),len(copy_trace))
 copies_by_key=defaultdict(list)
 for times,meta in zip(copy_rows,copy_trace):copies_by_key[meta['key']].append((mapped(times[1]),meta['source_event']))
 for stream in stream_ops:stream_ops[stream].sort()
 stream_starts={k:[a for a,b in v] for k,v in stream_ops.items()}
 events=defaultdict(lambda:dict(phases=defaultdict(list),details=[]))
 for serial,row in byid.items():
  event=row['event'];p=row['phase'];kr=kernels[serial];ints=[(a,b) for a,b,phase in kr if phase==p]
  detail=dict(row);detail['gpu_union_ms']=duration(ints)/1e6;detail['gpu_intervals_raw_ns']=[[mapped(a),mapped(b)] for a,b in ints]
  detail['api_intervals_raw_ns']=[[mapped(a),mapped(b)] for a,b in apis[serial]]
  events[event]['phases'][p].append(detail);events[event]['details'].append(detail)
 results=[]
 for event,data in sorted(events.items()):
  phases={}
  for phase,ds in data['phases'].items():
   ints=[tuple(i) for d in ds for i in d['gpu_intervals_raw_ns']];hs=[(d['host_start_raw_ns'],d['host_end_raw_ns']) for d in ds]
   phases[phase]=dict(host_union_ms=duration(hs)/1e6,gpu_union_ms=duration(ints)/1e6,host_start_ns=min(a for a,b in hs),host_end_ns=max(b for a,b in hs),gpu_start_ns=min((a for a,b in ints),default=None),gpu_end_ns=max((b for a,b in ints),default=None),count=len(ds))
  assert phases['forward_collective_host_call']['count']==phases['return_collective_host_call']['count']==1
  assert phases['forward_collective_host_call']['gpu_start_ns'] and phases['return_collective_host_call']['gpu_start_ns']
  loop=phases['expert_loop'];sub=['expert_ready_select','expert_gather','expert_compiled_kernel','expert_record_use','expert_weight_partial']
  loop['host_other_ms']=max(0,loop['host_union_ms']-sum(phases.get(p,{}).get('host_union_ms',0) for p in sub))
  eintervals=[tuple(i) for d in data['details'] if d['phase'] in sub for i in d['gpu_intervals_raw_ns']]
  loop['gpu_union_ms']=duration(eintervals)/1e6
  loop['gpu_span_ms']=(max(b for a,b in eintervals)-min(a for a,b in eintervals))/1e6
  loop['gpu_start_ns']=min(a for a,b in eintervals);loop['gpu_end_ns']=max(b for a,b in eintervals)
  # Ready_wait is nested inside ready_select. Never add both independently.
  loop['host_ready_select_without_wait_ms']=phases['expert_ready_select']['host_union_ms']-phases.get('expert_ready_wait',{}).get('host_union_ms',0)
  # Exclusive full-loop accounting on the aligned physical timeline. GPU
  # execution overlaps CPU enqueue/waits; priority partitions avoid adding it twice.
  start=loop['host_start_ns'];end=max(loop['host_end_ns'],loop['gpu_end_ns']);window=[(start,end)]
  compiled_ints=[tuple(i) for d in data['details'] if d['phase']=='expert_compiled_kernel' for i in d['gpu_intervals_raw_ns']]
  ready_ints=[(d['host_start_raw_ns'],d['host_end_raw_ns']) for d in data['details'] if d['phase']=='expert_ready_wait']
  wrapper_ints=[(d['host_start_raw_ns'],d['host_end_raw_ns']) for d in data['details'] if d['phase'] in sub]
  wrapper_ints += [tuple(i) for d in data['details'] if d['phase'] in ('expert_gather','expert_weight_partial') for i in d['gpu_intervals_raw_ns']]
  covered=[];account={}
  for name,ints in [('compiled_gpu',compiled_ints),('ready_wait',ready_ints),('wrapper',wrapper_ints),('loop_other',[(loop['host_start_ns'],loop['host_end_ns'])])]:
   ints=intersection(ints,window);exclusive=duration(ints)-duration(intersection(ints,covered));account[name+'_exclusive_ms']=exclusive/1e6;covered=union(covered+ints)
  account['full_loop_span_ms']=(end-start)/1e6;account['unexplained_ms']=((end-start)-duration(covered))/1e6
  assert abs(sum(account[k] for k in ('compiled_gpu_exclusive_ms','ready_wait_exclusive_ms','wrapper_exclusive_ms','loop_other_exclusive_ms','unexplained_ms'))-account['full_loop_span_ms'])<1e-6
  compiled_host=[(d['host_start_raw_ns'],d['host_end_raw_ns']) for d in data['details'] if d['phase']=='expert_compiled_kernel']
  account['compiled_host_without_gpu_ms']=(duration(compiled_host)-duration(intersection(compiled_host,compiled_ints)))/1e6
  wait_bounds=[]
  for d in data['details']:
   if d['phase']!='expert_ready_wait':continue
   gather=next(x for x in data['details'] if x['phase']=='expert_gather' and x['slot']==d['slot'] and x['key']==d['key'])
   assert gather['gpu_intervals_raw_ns']
   gs=min(a for a,b in gather['gpu_intervals_raw_ns']);candidates=[(end,source) for end,source in copies_by_key[d['key']] if source<=event and end<=gs]
   assert candidates,('missing slot DMA',event,d['key'])
   copy_end,source=max(candidates)
   stream=serial_stream[gather['serial']];ops=stream_ops[stream];starts=stream_starts[stream]
   raw_start=min(a for a,b,phase in kernels[gather['serial']]);i=bisect.bisect_left(starts,raw_start)-1
   prev_end=mapped(ops[i][1]) if i>=0 else d['host_start_raw_ns']
   upper=max(0,min(copy_end,gs)-max(prev_end,d['host_start_raw_ns']))/1e6
   wait_bounds.append(dict(slot=d['slot'],key=d['key'],source_event=source,dma_done_raw_ns=copy_end,gather_start_raw_ns=gs,gpu_wait_upper_bound_ms=upper,host_wait_ms=(d['host_end_raw_ns']-d['host_start_raw_ns'])/1e6))
  results.append(dict(event=event,rank=rank,step=event//48-1,layer=event%48,phases=phases,expert_exclusive_account=account,slot_wait_bounds=wait_bounds,expert_calls=[{k:d[k] for k in ('expert','rows','slot','key','gpu_union_ms','gpu_intervals_raw_ns','host_start_raw_ns','host_end_raw_ns')} for d in data['details'] if d['phase']=='expert_compiled_kernel'],ready_wait_calls=[d for d in data['details'] if d['phase']=='expert_ready_wait']))
 assert len(results)==384
 db.close();return dict(rank=rank,status='PASS',clock_alignment=clock,events=results,kernel_names=dict(names))
