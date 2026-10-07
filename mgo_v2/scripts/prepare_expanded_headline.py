"""Freeze owner-authorized 12-cell workload; never rewrite historical manifests."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006/expanded_matrix'
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 out=P/'WORKLOADS.json'
 assert not out.exists()
 source=Path('/home/hwlee/mgo-results/prefill_decode_policy_headroom_20261006/requests.json');pool=json.loads(source.read_text());assert pool['status']=='PASS'
 distinct=[];seen=set()
 for r in pool['requests']:
  if r['source_row'] not in seen:distinct.append(r);seen.add(r['source_row'])
 assert len(distinct)>=512
 cells=[]
 for c in [30,60]:
  slots=48*128*c//100;q,rem=divmod(slots,4)
  for b in [16,32,64]:
   for l in [256,512]:
    cell=f'R4_C{c}_B{b}_L{l}_O64';spec=dict(cell=cell,cache_percent=c,local_batch=b,global_requests=4*b,input_tokens=l,output_tokens=64,expert_slots=slots,expert_slots_per_rank=[q+(i<rem) for i in range(4)],expert_budget_bytes=slots*9*2**20)
    for phase,rows in [('target',distinct[:b*4]),('warmup',distinct[256:256+b*4])]:
     rr=[dict(request_id=r['request_id'],source_row=r['source_row'],conversation_id=r['conversation_id'],global_index=i,origin_rank=i//b,input_ids=r['input_ids'][-l:],input_tokens=l) for i,r in enumerate(rows)]
     assert all(len(r['input_ids'])==l for r in rr)
     f=P/'manifests'/f'{cell}_{phase}.json';write(f,dict(cell=cell,phase=phase,requests=rr));spec[phase]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    cells.append(spec)
 write(out,dict(status='FROZEN',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),physical_gpus=[0,1,4,5],primary_repeats=5,cells=cells))
 print(out)
if __name__=='__main__':main()
