"""CPU-only whole-ShareGPT first-eligible-prefix census for best64 search."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import sys
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np
from transformers import AutoTokenizer
from pcie_host import write

SOURCE=Path('/data2/esjung/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json')
SOURCE_SHA='35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4'
MODEL=Path('/data2/esjung/models/Qwen3-30B-A3B-Instruct-2507')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(out):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    assert sha(SOURCE)==SOURCE_SHA,'Frozen ShareGPT source hash mismatch'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    raw=json.loads(SOURCE.read_text());eligible=0;prefixes=0;lengths=[]
    unique=set();conversation_keys=set();duplicate=0;roles={'human':'user','gpt':'assistant','assistant':'assistant','system':'system'}
    legacy=Path('/data2/esjung/datasets/frozen_pcie_topology_20261009')
    known={}
    for phase in ('target','warmup'):
        for r in json.loads((legacy/f'R4_C30_B16_L512_O64_{phase}.json').read_text())['requests']:
            known[r['source_row']]=(phase,r['input_ids'])
    reproduced={};mismatch=[]
    with (out/'tokens.uint32').open('wb') as tokens,(out/'requests.jsonl').open('w') as meta:
        for source_row,conversation in enumerate(raw):
            messages=[]
            for turn,item in enumerate(conversation.get('conversations',[])):
                role=roles.get(item.get('from'));content=item.get('value')
                if role and isinstance(content,str) and content.strip():messages.append(dict(role=role,content=content))
                if item.get('from')!='human' or not isinstance(content,str) or not content.strip():continue
                ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                prefixes+=1
                if len(ids)<512:continue
                array=np.asarray(ids[-512:],np.uint32);digest=hashlib.sha256(array.tobytes()).hexdigest()
                duplicate+=int(digest in unique);unique.add(digest);lengths.append(len(ids))
                array.tofile(tokens)
                conversation_id=conversation.get('id')
                key=str(conversation_id).rsplit('_',1)[0] if conversation_id else f'row:{source_row}'
                conversation_keys.add(key)
                row=dict(pool_index=eligible,source_row=source_row,conversation_id=conversation_id,conversation_key=key,turn=turn,
                         original_token_count=len(ids),input_tokens=512,input_ids_uint32_sha256=digest)
                meta.write(json.dumps(row)+'\n');eligible+=1
                if source_row in known:
                    phase,expected=known[source_row]
                    if ids[-512:]==expected:reproduced[source_row]=phase
                    else:mismatch.append(dict(source_row=source_row,phase=phase))
                break
            if source_row%1000==0:
                write(out/'progress.json',dict(status='RUNNING',scanned_source_rows=source_row+1,total_source_rows=len(raw),
                        eligible_conversations=eligible,scanned_user_prefixes=prefixes,elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(scanned=source_row+1,total=len(raw),eligible=eligible)),flush=True)
    assert eligible>=64 and (out/'tokens.uint32').stat().st_size==eligible*512*4
    import torch
    assert not torch.cuda.is_initialized(),'CPU preparation initialized CUDA'
    result=dict(status='PASS',source_path=str(SOURCE),source_sha256=SOURCE_SHA,model=str(MODEL),
          selection='WHOLE corpus census: first user-ending conversation prefix >=512 Qwen chat-template tokens per source row, retain last512; no padding or duplication',
          scanned_source_rows=len(raw),eligible_conversations=eligible,unique_token_prefixes=len(unique),duplicate_token_prefixes=duplicate,
          distinct_conversation_families=len(conversation_keys),sample_unit='A distinct source-row request; split conversation families are recorded separately',
          scanned_user_prefixes=prefixes,input_tokens=512,token_dtype='little-endian uint32',tokens_shape=[eligible,512],
          original_token_min=min(lengths),original_token_max=max(lengths),original_token_mean=float(np.mean(lengths)),
          legacy_manifest_first_prefix_matches=len(reproduced),legacy_first_prefix_mismatches=mismatch,
          warmup_exclusion='Candidate selection must exclude frozen warmup source rows and exact token prefixes',
          source_identity_only=True,no_timing_selection_yet=True,cuda_initialized=False,wall_seconds=time.monotonic()-started,
          model_revision='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe',conda_prefix=sys.prefix,
          tokenizer_assets={name:sha(MODEL/name) for name in ('config.json','tokenizer.json','tokenizer_config.json')},
          artifacts={name:dict(bytes=(out/name).stat().st_size,sha256=sha(out/name)) for name in ('tokens.uint32','requests.jsonl')})
    write(out/'POOL.json',result);write(out/'progress.json',dict(status='PASS',eligible_conversations=eligible,scanned_source_rows=len(raw)))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
