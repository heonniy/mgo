"""Build a separately recorded long-prompt pool; never pad short requests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
from ttft_common import *
from transformers import AutoTokenizer

def main():
 target=ROOT/'requests.json'
 if target.exists():
  x=json.loads(target.read_text());assert x['status']=='PASS' and len(x['requests'])==2048;return
 old=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/ShareGPT_requests.json');receipt=json.loads(old.read_text());source=Path(receipt['source_path']);assert sha(source)==receipt['source_sha256']
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 raw=json.loads(source.read_text());rows=[];seen=set();scanned=0
 for i,conv in enumerate(raw):
  turns=conv.get('conversations',[])
  for turn,r in enumerate(turns):
   if r.get('from')!='human' or not isinstance(r.get('value'),str) or not r['value'].strip():continue
   prompt=r['value'];key=hashlib.sha256(prompt.encode()).hexdigest()
   if key in seen:continue
   seen.add(key);scanned+=1
   # 513 provides a strict length lower bound without tokenizing huge tails.
   ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True,enable_thinking=False,truncation=True,max_length=513)
   if len(ids)<512:continue
   ids=ids[:512]
   rows.append(dict(request_id=len(rows),source_row=i,turn=turn,prompt_sha256=key,input_ids=ids,input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),valid_tokens=512))
   if len(rows)%128==0:print('eligible',len(rows),'scanned',scanned,flush=True)
   if len(rows)==2048:break
  if len(rows)==2048:break
 assert len(rows)==2048,'insufficient distinct long requests; never duplicate/pad to fill pool'
 d=dict(status='PASS',source_path=str(source),source_sha256=receipt['source_sha256'],model=MODEL,requests=rows,selection='First 2048 distinct eligible human prompts in recorded corpus order; native chat tokenization, at least512 tokens then deterministic first512 truncation.',scanned_unique_prompts=scanned,context=512,pool_size=2048,masked_padding=False,decode_capture=False)
 write(target,d);write(PACKET/'TTFT_POOL.json',{k:v for k,v in d.items() if k!='requests'}|dict(manifest_path=str(target),manifest_sha256=sha(target)))
if __name__=='__main__':main()
