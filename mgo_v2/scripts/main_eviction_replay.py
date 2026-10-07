"""Deterministic MAIN-only eviction replay; per-event distinct LFU counts."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from numba import njit
POLICIES=('gate-score','LFU-reset','LFU-cumulative','LRU-reset','LRU-cumulative')

@njit(cache=True)
def replay(counts,gate_trace,capacities,mode,seed=42,layers=48):
 events,experts=counts.shape;world=len(capacities);keys=layers*experts
 slots=np.full((world,int(capacities.max())),-1,np.int32)
 owner=np.full(keys,-1,np.int32);freq=np.zeros(keys,np.int64);last=np.zeros(keys,np.int64)
 gates=np.zeros(keys,np.float32);ever=np.zeros(keys,np.bool_);metrics=np.zeros((events,4),np.int64)
 np.random.seed(seed)
 for event in range(events):
  layer=event%layers;active=counts[event]>0;gates[layer*experts:(layer+1)*experts]=gate_trace[event]
  misses=np.empty(experts,np.int64);n=0
  for e in range(experts):
   if active[e]:
    key=layer*experts+e
    if owner[key]>=0:metrics[event,0]+=1
    else:misses[n]=e;n+=1
  metrics[event,1]=n
  # Same BR quota and shuffle implementation as balanced_assignment.
  quota=np.empty(n,np.int64);at=0
  for rank in range(world):
   for _ in range(n//world+int(rank<n%world)):quota[at]=rank;at+=1
  order=np.arange(n);np.random.shuffle(order);assignment=np.empty(n,np.int64)
  for j in range(n):assignment[order[j]]=quota[j]
  for j in range(n):
   expert=misses[j];key=layer*experts+expert;rank=assignment[j]
   best=-1;best_score=np.inf;best_last=np.iinfo(np.int64).max;best_key=np.iinfo(np.int64).max
   for slot in range(capacities[rank]):
    victim=slots[rank,slot]
    if victim<0:best=slot;break
    if victim//experts==layer and active[victim%experts]:continue
    score=float(gates[victim]) if mode==0 else (float(freq[victim]) if mode<=2 else 0.)
    used=last[victim]
    if score<best_score or (score==best_score and (used<best_last or (used==best_last and victim<best_key))):
     best=slot;best_score=score;best_last=used;best_key=victim
   if best<0:raise ValueError('MAIN capacity cannot admit active experts under BR quota')
   victim=slots[rank,best]
   if victim>=0:
    metrics[event,2]+=1;owner[victim]=-1
    if mode==1 or mode==3:freq[victim]=0;last[victim]=0
   if ever[key]:metrics[event,3]+=1
   ever[key]=True;slots[rank,best]=key;owner[key]=rank
  for e in range(experts):
   if active[e]:
    key=layer*experts+e;freq[key]+=1;last[key]=event+1
 return metrics,slots,freq,last

def run(trace,output):
 receipt=json.loads((trace.parent/'trace_receipt.json').read_text())
 assert receipt['status']=='PASS' and receipt['decode_forwards']==256 and not receipt['prefetch']
 with np.load(trace,allow_pickle=False) as data:
  counts=data['counts'];gates=data['gates'];reference=data['reference_metrics'];final_slots=data['final_slots']
 assert counts.shape==(257*48,128) and gates.shape==counts.shape
 assert counts.dtype==np.int32 and gates.dtype==np.float32 and np.all(counts>=0) and np.isfinite(gates).all()
 assert np.all(counts[48:].sum(1)==receipt['local_batch']*4*8)
 content=hashlib.sha256(counts.tobytes()+gates.tobytes()+reference.tobytes()).hexdigest();assert content==receipt['content_sha256']
 capacities=np.array(receipt['capacities'],np.int32);output.mkdir(parents=True,exist_ok=True);rows=[];all_metrics=[];all_slots=[]
 for mode,name in enumerate(POLICIES):
  metrics,slots,_,_=replay(counts,gates,capacities,mode,receipt['seed'])
  assert np.array_equal(metrics[:,:2].sum(1),(counts>0).sum(1))
  assert np.all(metrics[:,3]<=metrics[:,1]) and np.all(metrics[:,2]<=metrics[:,1])
  assert int(metrics[:,1].sum()-metrics[:,2].sum())==int((slots>=0).sum())
  if mode==0:
   assert np.array_equal(metrics,reference),'Gate source/replay event count mismatch'
   assert np.array_equal(slots,final_slots),'Gate source/replay final cache mismatch'
  per_step=metrics.reshape(257,48,4).sum(1);all_metrics.append(metrics);all_slots.append(slots)
  fields=('hit','miss','eviction','reload');summary=dict(policy=name,prefill=dict(zip(fields,map(int,per_step[0]))),decode=dict(zip(fields,map(int,per_step[1:].sum(0)))),checkpoints={})
  summary['decode']['hit_rate']=summary['decode']['hit']/(summary['decode']['hit']+summary['decode']['miss'])
  with (output/f'{name}_steps.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['step',*fields,*['cumulative_'+k for k in fields]])
   cum=np.zeros(4,np.int64)
   for step,v in enumerate(per_step):
    if step:cum+=v
    writer.writerow([step,*v,*cum])
    if step in (1,8,16,32,64,128,256):summary['checkpoints'][str(step)]=dict(zip(fields,map(int,cum)))
  rows.append(summary)
 assert np.array_equal(all_metrics[3],all_metrics[4]) and np.array_equal(all_slots[3],all_slots[4]),'LRU reset/cumulative must be equivalent'
 result=dict(status='PASS',local_batch=receipt['local_batch'],trace=str(trace),trace_content_sha256=content,prefetch=False,capacity=sum(receipt['capacities']),gate_reference_parity=True,lru_equivalence=True,policies=rows)
 (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.trace,a.output),indent=2))
