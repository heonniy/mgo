"""Build a separately recorded long-prompt pool; never pad short requests."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
from ttft_common import *
from transformers import AutoTokenizer

def main():
 target=ROOT/'requests.json'
 if target.exists():
  x=json.loads(target.read_text());assert x['status']=='PASS' and len(x['requests'])==2048 and x['truncation']=='last512_formatted_conversation';return
 old=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004/ShareGPT_requests.json');receipt=json.loads(old.read_text());source=Path(receipt['source_path']);assert sha(source)==receipt['source_sha256']
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 raw=json.loads(source.read_text());rows=[];seen=set();scanned=0;lengths=[]
 for i,conv in enumerate(raw):
  turns=conv.get('conversations',[]);messages=[]
  for turn,r in enumerate(turns):
   role={'human':'user','gpt':'assistant','assistant':'assistant','system':'system'}.get(r.get('from'))
   if role and isinstance(r.get('value'),str) and r['value'].strip():messages.append(dict(role=role,content=r['value']))
   if r.get('from')!='human' or not isinstance(r.get('value'),str) or not r['value'].strip():continue
   ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
   original=len(ids);scanned+=1;lengths.append(original)
   if original<512:continue
   ids=ids[-512:];key=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
   if key in seen:continue
   seen.add(key)
   rows.append(dict(request_id=len(rows),source_row=i,conversation_id=conv.get('id'),turn=turn,conversation_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest(),original_token_count=original,input_ids=ids,input_ids_sha256=key,valid_tokens=512))
   if len(rows)%128==0:print('eligible',len(rows),'scanned',scanned,flush=True)
   if len(rows)==2048:break
  if len(rows)==2048:break
 assert len(rows)==2048,'insufficient distinct long requests; never duplicate/pad to fill pool'
 d=dict(status='PASS',source_path=str(source),source_sha256=receipt['source_sha256'],model=MODEL,requests=rows,selection='First 2048 distinct eligible conversation prefixes ending in a user turn, corpus order; same Qwen chat template, retain last512 formatted tokens.',truncation='last512_formatted_conversation',scanned_conversation_prefixes=scanned,eligible_count=2048,eligible_count_scope='Distinct eligible inputs retained from scanned corpus prefix; not a full-corpus census.',original_pool_token_min=min(r['original_token_count'] for r in rows),original_pool_token_max=max(r['original_token_count'] for r in rows),original_pool_token_mean=sum(r['original_token_count'] for r in rows)/len(rows),context=512,pool_size=2048,masked_padding=False,decode_capture=False)
 write(target,d);write(PACKET/'TTFT_POOL.json',{k:v for k,v in d.items() if k!='requests'}|dict(manifest_path=str(target),manifest_sha256=sha(target)))
if __name__=='__main__':main()
