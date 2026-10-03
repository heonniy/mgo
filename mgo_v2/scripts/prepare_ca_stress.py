"""Extend request manifests under original filters; never recapture the prefix."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
from collections import defaultdict
import random,hashlib,json
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from ca_stress_common import *

def render(tok,text):return tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
def main():
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 for name in ['MATH','ShareGPT']:
  output=ROOT/(name+'_requests.json')
  if output.exists():assert json.loads(output.read_text())['status']=='PASS';continue
  original=json.loads((SOURCE/(name+'_requests.json')).read_text());source=Path(original['source_path']);assert sha(source)==original['source_sha256'];prefix=original['requests'];assert len(prefix)==512
  old_hashes={r['prompt_sha256'] for r in prefix};extras=[]
  if name=='MATH':
   groups=defaultdict(list);old_rows={r['source_row'] for r in prefix};raw=pq.read_table(source).to_pylist();eligible=len(raw)
   for i,r in enumerate(raw):
    if i not in old_rows:groups[r['type'],r['level']].append(dict(source_row=i,prompt=r['problem'],subject=r['type'],difficulty=r['level']))
   rng=random.Random(42)
   for key in sorted(groups):rng.shuffle(groups[key])
   target=min(1536,sum(map(len,groups.values())))
   while len(extras)<target:
    for key in sorted(groups):
     if groups[key] and len(extras)<target:extras.append(groups[key].pop())
   rng.shuffle(extras)
   for r in extras:
    ids=render(tok,r['prompt']);r.update(input_ids=ids[:512],untruncated_input_tokens=len(ids),truncated=len(ids)>512)
   selection='preserve original512; remaining source rows, seed42 balanced subject/difficulty round robin then shuffle'
  else:
   raw=json.loads(source.read_text());seen=set();eligible_rows=[];batch=[]
   def consume(items):
    if not items:return
    prompts=tok([r['prompt'] for r in items],add_special_tokens=False)['input_ids'];answers=tok([r.pop('answer') for r in items],add_special_tokens=False)['input_ids']
    for r,ids,answer in zip(items,prompts,answers):
     if not 32<=len(ids)<=512 or len(answer)<128:continue
     chat=render(tok,r['prompt'])
     if len(chat)>512:continue
     r.update(input_ids=chat,prompt_tokens=len(ids),reference_tokens=len(answer),truncated=False);eligible_rows.append(r)
   for row,conv in enumerate(raw):
    turns=conv.get('conversations',[])
    for turn in range(len(turns)-1):
     a,b=turns[turn:turn+2]
     if a.get('from')!='human' or b.get('from') not in ('gpt','assistant'):continue
     prompt=a.get('value');answer=b.get('value')
     if not isinstance(prompt,str) or not isinstance(answer,str) or not prompt.strip() or not answer.strip():continue
     key=hashlib.sha256(prompt.encode()).hexdigest()
     if key in seen:continue
     seen.add(key);batch.append(dict(source_row=row,turn=turn,conversation_id=conv.get('id'),prompt=prompt,answer=answer,reference_sha256=hashlib.sha256(answer.encode()).hexdigest()))
     if len(batch)>=256:consume(batch);batch=[]
   consume(batch);eligible=len(eligible_rows);assert eligible==original['eligible'],(eligible,original['eligible'])
   eligible_by_hash={hashlib.sha256(r['prompt'].encode()).hexdigest():r for r in eligible_rows}
   assert all(eligible_by_hash[r['prompt_sha256']]['input_ids']==r['input_ids'] for r in prefix)
   remaining=[r for h,r in eligible_by_hash.items() if h not in old_hashes];extras=random.Random(44).sample(remaining,min(1536,len(remaining)))
   selection='preserve original512; seed44 uniform sample without replacement from remaining original-filter eligible deduplicated turns'
  for i,r in enumerate(extras,512):r.update(request_id=i,prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),input_ids_sha256=digest(r['input_ids']),logical_rank=i%8)
  rows=prefix+extras;assert len(rows)<=2048 and all(0<len(r['input_ids'])<=512 for r in rows)
  assert rows[:512]==prefix and len({r['request_id'] for r in rows})==len(rows)
  obj=dict(status='PASS',dataset=name,pool_size=len(rows),eligible=eligible,original_prefix=512,additional_capture=len(extras),selection=selection,model_path=str(MODEL),source_path=str(source),source_sha256=original['source_sha256'],original_manifest_sha256=sha(SOURCE/(name+'_requests.json')),requests=rows)
  write(output,obj)
  for wave,start in enumerate(range(512,len(rows),512)):
   write(ROOT/'waves'/f'wave{wave}'/(name+'_requests.json'),dict(status='PASS',parent_manifest_sha256=sha(output),requests=rows[start:start+512]))
  write(PACKET/(name+'_manifest.json'),{**{k:v for k,v in obj.items() if k!='requests'},'manifest_sha256':sha(output),'manifest_path':str(output),'requests':[{k:v for k,v in r.items() if k not in ['prompt','input_ids']} for r in rows]})
  print(name,len(rows),'frozen; original512 unchanged',flush=True)
if __name__=='__main__':main()
