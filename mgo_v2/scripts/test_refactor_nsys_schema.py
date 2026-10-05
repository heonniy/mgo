"""Synthetic schema test; does not replace a physical Nsight capture gate."""
import hashlib,json,sqlite3,tempfile
from pathlib import Path
from analyze_refactor_nsys import analyze

def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);path=root/'trace.sqlite';db=sqlite3.connect(path)
  for sql in [
   'CREATE TABLE StringIds(id INTEGER,value TEXT)',
   'CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER)',
   'CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER)',
   'CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER)',
   'CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,demangledName INTEGER)',
   'CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,label TEXT)',
   'CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER)']:db.execute(sql)
  db.executemany('INSERT INTO StringIds VALUES (?,?)',[(1,'ncclDevKernel'),(2,'triton_expert')])
  db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (1,'HtoD')")
  pid=1<<24;tid=pid+1
  phases=[('moe.metadata',1,9,3,7,1),('moe.forward_a2a',10,25,12,25,1),('moe.expert_compute',26,60,30,55,2),('moe.return_a2a',61,90,65,80,1)]
  for event in range(384):
   base=1000+event*100
   db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)',(base,base+99,tid,f'decode.event.{event+48}'))
   for phase,(name,a,b,c,d,kernel) in enumerate(phases):
    corr=event*4+phase
    db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)',(base+a,base+b,tid,name))
    if name=='moe.metadata':
     db.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,NULL)',(base+a,base+a+1,tid,'moe.metadata_collective_submit'))
    # Exercise direct-driver attribution for the compute launch.
    api='DRIVER' if kernel==2 else 'RUNTIME'
    db.execute(f'INSERT INTO CUPTI_ACTIVITY_KIND_{api} VALUES (?,?,?,?)',(base+a,base+a+1,tid,corr))
    db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?)',(base+c,base+d,pid,corr,kernel))
  db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (1010,1040,1,9437184)');db.commit();db.close()
  trace=root/'copy.json';trace.write_text(json.dumps([dict(source_event=48,bytes=9437184,kind='useful_prefetch',readiness_at_use='inflight')]))
  receipt=dict(status='PASS',case=dict(horizon=8),decode_expert_copies=1,decode_expert_bytes=9437184,copy_trace_path=str(trace),copy_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
  out=analyze(path,receipt)
  assert out['decode_events']==384 and out['kernel_counts']['moe.expert_compute']==384
  assert out['H2D']['total_union_ms']==30/1e6
  assert out['H2D']['overlap_comm_expert_ms']==23/1e6
  inflight=out['H2D_split']['readiness_at_use']['inflight']
  assert inflight['count']==1 and inflight['bytes']==9437184
  assert inflight['total_union_ms']==30/1e6 and inflight['overlap_comm_expert_ms']==23/1e6
  assert len(out['interval_decomposition']['per_event'])==384
  causal=out['interval_decomposition']['causal_event_kernel_union_ms']
  assert len(causal)==384 and causal['48']['moe.forward_a2a.nccl']==13/1e6
  forward=out['interval_decomposition']['causal_event_kernel_distributions']['moe.forward_a2a.nccl']
  assert forward['median_ms']==13/1e6 and abs(forward['p90_ms']-13/1e6)<1e-15
  preview=out['interval_decomposition']['timeline_preview']
  assert preview['window_ns']==(1000,1300)
  assert preview['intervals_ns']['expert_H2D_DMA']==[(1010,1040)]
  assert len(preview['intervals_ns']['moe.expert_compute'])==3
  cpu=out['cpu_nvtx']
  metadata_parts=sum(cpu[name]['total_ms'] for name in ('moe.metadata_pack_host','moe.metadata_collective_submit','moe.metadata_readback_wait_and_unpack'))
  assert abs(metadata_parts-cpu['moe.metadata']['total_ms'])<1e-12
  receipt['case']['horizon']=256
  try:analyze(path,receipt)
  except AssertionError as exc:assert 'wrong captured horizon' in str(exc)
  else:raise AssertionError('short capture must not masquerade as full-horizon evidence')
  receipt['case']['horizon']=8
  receipt['decode_expert_copies']=2
  try:analyze(path,receipt)
  except AssertionError as exc:assert 'copy count mismatch' in str(exc)
  else:raise AssertionError('missing DMA event must fail reconciliation')
 print('PASS schema fixture: runtime/driver correlation, DMA overlap, per-event conservation, missing-copy rejection')

if __name__=='__main__':main()
