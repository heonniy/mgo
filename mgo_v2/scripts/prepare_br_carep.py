#!/usr/bin/env python3
"""Pin workload selection and verify the reusable SERE calibration, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,random
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
P=Path(__file__).resolve().parents[1]/'experiments/br_ca_carep_cpu_headroom_20261003'
ROOT=Path('/home/hwlee/mgo-results/br_ca_carep_cpu_headroom_20261003')
MODEL=Path('/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507')
OLD=Path('/home/hwlee/sub-moe-results/substitution_characterization_20260928_r1')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def audit(tok):
    manifest=json.loads((OLD/'input_manifest.json').read_text());cal=np.load(OLD/'inputs.npz')['calibration']
    assert cal.shape==(400,128) and manifest['seed']==42
    assert sha(OLD/'inputs.npz')==manifest['inputs_sha256']
    revision=(MODEL/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0]
    assert revision==manifest['revisions']['model']=='0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe'
    fine=OLD/'input/sample/10BT/000_00000.parquet';assert sha(fine)==manifest['source_files'][str(fine)]
    selected=[r for r in manifest['samples'] if r['group']=='calibration'];assert len(selected)==400
    wanted={r['source_row'] for r in selected};data={};offset=0
    for batch in pq.ParquetFile(fine).iter_batches(batch_size=1024,columns=['text']):
        for i,row in enumerate(batch.to_pylist()):
            if offset+i in wanted:data[offset+i]=row
        offset+=len(batch)
        if len(data)==len(wanted):break
    for i,row in enumerate(selected):
        text=data[row['source_row']]['text'];assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256']
        ids=tok(text,add_special_tokens=True,truncation=True,max_length=128)['input_ids'];assert np.array_equal(ids,cal[i])
    sim=np.load(OLD/'similarities.npy');current=Path('/home/hwlee/mgo-results/runtime_validation_20261001/similarity.npy');assert np.array_equal(sim,np.load(current))
    ref=Path('/home/hwlee/sub-moe/reference/SERE/code/calibration/utils.py');layers=[];receipts=[]
    for lane in range(4):
        d=OLD/f'c2_lane{lane}';receipt=json.loads((d/'receipt.json').read_text());env=json.loads((d/'environment.json').read_text())
        assert receipt['status']=='COMPLETE' and not receipt['failures']
        assert receipt['reference_source_sha256']==sha(ref) and 'BF16(Y_i-Y_j)' in receipt['normalization']
        assert env['revisions']['model']==revision and env['input_manifest_sha256']==sha(OLD/'input_manifest.json')
        for r in receipt['layers']:
            layer=r['layer'];f=d/f'layer_{layer:02d}.npy';assert r['samples']==400 and r['tokens']==51200 and sha(f)==r['sha256'];assert np.array_equal(np.load(f),sim[layer]);layers.append(layer)
        receipts.append(dict(path=str(d/'receipt.json'),sha256=sha(d/'receipt.json'),environment_sha256=sha(d/'environment.json')))
    assert sorted(layers)==list(range(48)) and np.isfinite(sim).all()
    out=dict(status='REUSE_VERIFIED',model_revision=revision,source_dataset='HuggingFaceFW/fineweb-edu',source_revision=manifest['revisions']['fineweb'],seed=42,selection=manifest['selection'],sequences=400,tokens_per_sequence=128,source_text_and_tokens_verified=400,layer_matrices_verified=48,method='SERE Frobenius output similarity',normalization=receipt['normalization'],similarity_path=str(current),similarity_sha256=sha(current),original_similarity_sha256=sha(OLD/'similarities.npy'),reference_path=str(ref),reference_sha256=sha(ref),input_manifest_path=str(OLD/'input_manifest.json'),input_manifest_sha256=sha(OLD/'input_manifest.json'),fineweb_path=str(fine),fineweb_sha256=sha(fine),lane_receipts=receipts,recalibration_runs=0)
    write(P/'similarity_audit.json',out);print('similarity REUSE_VERIFIED',flush=True)
def render(tok,text):
    return tok.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
def main():
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);audit(tok)
    mathfile=ROOT/'sources/MATH-lighteval/data/test-00000-of-00001.parquet';groups=defaultdict(list)
    for i,r in enumerate(pq.read_table(mathfile).to_pylist()):
        groups[r['type'],r['level']].append(dict(source_row=i,prompt=r['problem'],subject=r['type'],difficulty=r['level']))
    rng=random.Random(42)
    for key in sorted(groups):rng.shuffle(groups[key])
    chosen=[]
    while len(chosen)<512:
        for key in sorted(groups):
            if groups[key] and len(chosen)<512:chosen.append(groups[key].pop())
    rng.shuffle(chosen)
    for r in chosen:
        ids=render(tok,r['prompt']);r.update(input_ids=ids[:512],untruncated_input_tokens=len(ids),truncated=len(ids)>512)
    mathmeta=dict(dataset='DigitalLearningGmbH/MATH-lighteval',revision='0530c78699ea5e8eb5530600900e1f328b48acad',split='test',seed=42,selection='shuffle within subject/difficulty, balanced round-robin quotas, shuffle final order',strata=dict(Counter(r['subject']+' / '+r['difficulty'] for r in chosen)))
    save('MATH',chosen,mathmeta,mathfile)
    sharefile=ROOT/'sources/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json';raw=json.loads(sharefile.read_text());eligible=[];seen=set();stats=Counter()
    for row,conv in enumerate(raw):
        turns=conv.get('conversations',[])
        for turn in range(len(turns)-1):
            a,b=turns[turn:turn+2]
            if a.get('from')!='human' or b.get('from') not in ('gpt','assistant'):continue
            prompt=a.get('value');answer=b.get('value')
            if not isinstance(prompt,str) or not isinstance(answer,str) or not prompt.strip() or not answer.strip():stats['malformed']+=1;continue
            key=hashlib.sha256(prompt.encode()).hexdigest()
            if key in seen:stats['duplicate_prompt']+=1;continue
            seen.add(key);ids=tok(prompt,add_special_tokens=False)['input_ids']
            if not 32<=len(ids)<=512:stats['prompt_length']+=1;continue
            ref=tok(answer,add_special_tokens=False)['input_ids']
            if len(ref)<128:stats['short_reference']+=1;continue
            chat=render(tok,prompt)
            # Preserve the full selected user turn, including template overhead.
            if len(chat)>512:stats['chat_over_512']+=1;continue
            eligible.append(dict(source_row=row,turn=turn,conversation_id=conv.get('id'),prompt=prompt,input_ids=chat,prompt_tokens=len(ids),reference_tokens=len(ref),reference_sha256=hashlib.sha256(answer.encode()).hexdigest(),truncated=False))
        if row%10000==0:print('ShareGPT scanned',row,'eligible',len(eligible),flush=True)
    assert len(eligible)>=512
    chosen=random.Random(44).sample(eligible,512)
    meta=dict(dataset='anon8231489123/ShareGPT_Vicuna_unfiltered',revision='192ab2185289094fc556ec8ce5ce1e8e587154ca',source_file=sharefile.name,seed=44,selection='uniform sample of eligible deduplicated human->assistant turns; preserve full user prompt',eligible=len(eligible),rejections=dict(stats))
    save('ShareGPT',chosen,meta,sharefile)
def save(name,rows,meta,source):
    for i,r in enumerate(rows):r.update(request_id=i,prompt_sha256=hashlib.sha256(r['prompt'].encode()).hexdigest(),input_ids_sha256=digest(r['input_ids']),logical_rank=i%8)
    assert len(rows)==512 and len({r['prompt_sha256'] for r in rows})==512 and all(0<len(r['input_ids'])<=512 for r in rows)
    meta.update(source_path=str(source),source_sha256=sha(source),model_path=str(MODEL),samples=512,capture_decode_steps=256,CPU_horizons=[64,256],ordering_sha256=digest([r['input_ids_sha256'] for r in rows]),chat_template='Qwen user-only; add_generation_prompt=True; enable_thinking=False',input_truncation='MATH right truncation to 512; ShareGPT selected rendered prompt <=512 without truncation')
    dest=ROOT/(name+'_requests.json');assert not dest.exists();write(dest,dict(**meta,requests=rows))
    write(P/(name+'_manifest.json'),dict(**meta,requests_path=str(dest),requests_sha256=sha(dest),requests=[{k:v for k,v in r.items() if k not in ('prompt','input_ids')} for r in rows]));print(name,'512 frozen',flush=True)
if __name__=='__main__':main()
