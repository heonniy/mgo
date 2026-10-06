"""Freeze shared non-adversarial global workloads before any system timing."""
from pathlib import Path
import hashlib,json
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
PACKET=Path(__file__).resolve().parents[1]/'experiments/main_table_global_workload_20261006'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':
 assert not (ROOT/'WORKLOADS.json').exists()
 source=Path('/home/hwlee/mgo-results/prefill_decode_policy_headroom_20261006/requests.json');pool=json.loads(source.read_text());assert pool['status']=='PASS'
 distinct=[];seen=set()
 for r in pool['requests']:
  if r['source_row'] not in seen:distinct.append(r);seen.add(r['source_row'])
 assert len(distinct)>=512
 manifest=[]
 for batch,length in [(16,256),(64,512)]:
  n=batch*4;cell=f'R4_C30_B{batch}_L{length}_O64';phases={}
  for phase,rows in [('target',distinct[:n]),('warmup',distinct[256:256+n])]:
   out=[]
   for i,r in enumerate(rows):
    ids=r['input_ids'][-length:];assert len(ids)==length
    out.append(dict(request_id=r['request_id'],source_row=r['source_row'],conversation_id=r['conversation_id'],global_index=i,origin_rank=i//batch,input_ids=ids,input_tokens=length))
   path=ROOT/'manifests'/f'{cell}_{phase}.json';write(path,dict(cell=cell,phase=phase,global_requests=n,input_tokens=length,output_tokens=64,requests=out));phases[phase]=dict(path=str(path),sha256=sha(path))
  assert not {r['source_row'] for r in distinct[:n]}&{r['source_row'] for r in distinct[256:256+n]}
  manifest.append(dict(cell=cell,local_batch=batch,global_requests=n,input_tokens=length,output_tokens=64,**phases))
 result=dict(status='FROZEN',source=str(source),source_sha256=sha(source),selection='First eligible request per distinct source conversation in frozen pool order. Target conversations0:256; warmup256:512. Smaller cell uses first64 of each. No policy search or timing selection.',physical_gpus=[0,1,4,5],systems=['llama.cpp-layer','DeepSpeed ZeRO-Inference','MoE-Infinity','Ours'],ours_policy='LA_CA_NEAR',repeats=3,cells=manifest)
 write(ROOT/'WORKLOADS.json',result);write(PACKET/'HEADLINE_R4_WORKLOADS.json',result)
 print('PASS: two cells; identical token IDs across systems; disjoint warmup conversations')
