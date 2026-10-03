#!/usr/bin/env python3
"""One resident exact Qwen load per GPU, two sequential master-trace shards."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,inspect,json,resource,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
ROOT=Path('/home/hwlee/mgo-results/ca_stress_workload_search_20261004')
BASE_ROOT=ROOT
MODEL='/home/hwlee/model/Qwen3-30B-A3B-Instruct-2507'
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def check_memory():
    free,total=torch.cuda.mem_get_info()
    available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    if free<8*2**30 or available<256*2**30:raise RuntimeError(f'Memory guard: GPU free={free}, host available={available}')
    return dict(gpu_free_bytes=free,host_available_bytes=available,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
class Trace:
    def __init__(self,path,mask,request_ids):
        path.mkdir(parents=True,exist_ok=False);self.path=path;self.step=0;self.layer=0;self.mask=mask.reshape(-1).bool();self.request_ids=request_ids
        n=int(mask.sum());self.arrays={}
        for scope,tokens in (('prefill',n),('decode',len(request_ids))):
            for name,width,dtype in (('selected',8,'uint8'),('weights',8,'float32'),('router',128,'float32')):
                shape=(48,tokens,width) if scope=='prefill' else (256,48,tokens,width)
                self.arrays[scope+'_'+name]=np.lib.format.open_memmap(path/(scope+'_'+name+'.npy'),mode='w+',dtype=dtype,shape=shape)
        np.save(path/'prefill_request_ids.npy',np.repeat(np.array(request_ids,dtype=np.int32),mask.sum(1).cpu().numpy()))
        np.save(path/'prefill_positions.npy',np.concatenate([np.arange(int(n),dtype=np.int32) for n in mask.sum(1).cpu().tolist()]))
    def hook(self,layer):
        def capture(module,args,logits):
            assert layer==self.layer
            probs=torch.softmax(logits,dim=-1,dtype=torch.float32)
            weights,selected=torch.topk(probs,8,dim=-1);weights=(weights/weights.sum(-1,keepdim=True)).to(logits.dtype).float()
            if self.step==0:
                probs=probs[self.mask];weights=weights[self.mask];selected=selected[self.mask];key='prefill';index=layer
            else:key='decode';index=(self.step-1,layer)
            for name,value in (('selected',selected),('weights',weights),('router',probs)):
                a=value.detach().cpu().numpy();assert np.isfinite(a).all();self.arrays[key+'_'+name][index]=a
            self.layer+=1
        return capture
    def done_forward(self):
        assert self.layer==48;self.layer=0;self.step+=1
    def flush(self):
        for a in self.arrays.values():a.flush()
def capture(model,rank,name):
    source=ROOT/(name+'_requests.json');manifest=json.loads(source.read_text());rows=manifest['requests'][rank::8];assert len(rows)<=64 and all(r['request_id']>=512 for r in rows)
    if not rows:
        path=ROOT/'captures'/name/f'rank{rank}';path.mkdir(parents=True,exist_ok=False);write(path/'receipt.json',dict(status='PASS',request_ids=[],files=[],decode_forwards=256));return
    length=max(len(r['input_ids']) for r in rows);pad=model.generation_config.pad_token_id or model.config.eos_token_id
    ids=torch.full((len(rows),length),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
    for i,r in enumerate(rows):ids[i,-len(r['input_ids']):]=torch.tensor(r['input_ids'],device='cuda');mask[i,-len(r['input_ids']):]=1
    path=ROOT/'captures'/name/f'rank{rank}';trace=Trace(path,mask,[r['request_id'] for r in rows]);hooks=[layer.mlp.gate.register_forward_hook(trace.hook(i)) for i,layer in enumerate(model.model.layers)]
    np.save(path/'prefill_input_ids.npy',ids.cpu().numpy());np.save(path/'prefill_active_mask.npy',mask.cpu().numpy())
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    past=None;generated=[]
    try:
        with torch.inference_mode():
            # One prefill generates g1; 256 decode forwards consume g1..g256.
            # The final logits are discarded: exactly 256 generated tokens.
            for step in range(257):
                check_memory();position=mask.cumsum(-1)-1;position.masked_fill_(mask==0,0)
                out=model(input_ids=ids,attention_mask=mask,position_ids=position[:,-ids.shape[1]:],past_key_values=past,use_cache=True,logits_to_keep=1)
                trace.done_forward();past=out.past_key_values
                if step<256:
                    logits=out.logits[:,-1].clone();logits[:,eos]=-torch.inf
                    ids=logits.argmax(-1,keepdim=True);generated.append(ids.cpu().numpy());mask=torch.cat((mask,mask.new_ones((len(rows),1))),1)
                    del logits
                del out
                if step%16==0 or step==256:
                    trace.flush();state=dict(status='RUNNING',dataset=name,rank=rank,decode_completed=step,generated_tokens=len(generated),**check_memory());write(ROOT/f'gpu{rank}_progress.json',state);print(json.dumps(state),flush=True)
        assert trace.step==257 and len(generated)==256
        np.save(path/'generated_tokens.npy',np.concatenate(generated,axis=1));trace.flush()
        receipts=[dict(file=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(path.glob('*.npy'))]
        write(path/'receipt.json',dict(status='PASS',dataset=name,rank=rank,request_ids=trace.request_ids,request_manifest_sha256=sha(source),prefill_forwards=1,decode_forwards=256,generated_tokens_per_request=256,active_decode_tokens=len(rows)*256,model_loads_this_worker=1,substitution=False,full_router_dtype='float32',selected_weight_dtype='native BF16 converted losslessly to float32',prefix64='decode arrays [:64], generated_tokens [:,:64]',files=receipts,**check_memory()))
    finally:
        for hook in hooks:hook.remove()
    del past,trace,ids,mask,generated;torch.cuda.empty_cache()
def wait_file(path,timeout=7200):
    started=time.monotonic()
    while not path.exists():
        if (ROOT/'STOP').exists():raise RuntimeError('Requested stop')
        if time.monotonic()-started>timeout:raise TimeoutError(str(path))
        time.sleep(1)
def main(rank):
    global ROOT
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(rank)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.88);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    check_memory()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    write(BASE_ROOT/f'gpu{rank}_loaded.json',dict(pid=os.getpid(),rank=rank,model=MODEL,modeling_source=inspect.getfile(type(model)),modeling_sha256=sha(inspect.getfile(type(model))),**check_memory()))
    for name in ['MATH','ShareGPT']:
        for wave in sorted((BASE_ROOT/'waves').glob('wave*')):
            if (BASE_ROOT/'STOP').exists():raise RuntimeError('Owner stop')
            if not (wave/(name+'_requests.json')).exists():continue
            ROOT=wave;capture(model,rank,name)
    write(BASE_ROOT/f'gpu{rank}_capture_complete.json',dict(status='PASS',rank=rank,model_loads=1,recaptured_prefix_requests=0))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int,required=True);main(parser.parse_args().rank)
