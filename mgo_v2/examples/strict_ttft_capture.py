"""Exact-model ShareGPT_LONG256 route capture using physical GPUs 0,1,4,5 only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,json,resource,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from strict_headroom_common import ROOT,OLD_ROOT,MODEL,R4_GPUS,write,sha,request_rows

CONTEXT=256
CAPTURE_BATCH=8

def memory():
    free,total=torch.cuda.mem_get_info()
    host=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    if free<8*2**30 or host<256*2**30:raise RuntimeError(dict(gpu_free=free,host_available=host))
    return dict(gpu_free_bytes=free,host_available_bytes=host,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)

def main(a):
    assert a.gpu in R4_GPUS and os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    all_rows=request_rows();i=R4_GPUS.index(a.gpu);rows=all_rows[i*512:(i+1)*512];assert len(rows)==512
    out=ROOT/'capture_L256'/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False);memory()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
    assert len(model.model.layers)==48 and model.config.num_experts==128
    arrays={
      'selected':np.lib.format.open_memmap(out/'selected.npy',mode='w+',dtype='uint8',shape=(512,48,CONTEXT,8)),
      'weights':np.lib.format.open_memmap(out/'weights.npy',mode='w+',dtype='float32',shape=(512,48,CONTEXT,8)),
      'gates':np.lib.format.open_memmap(out/'gates.npy',mode='w+',dtype='float32',shape=(512,48,128)),
    }
    generated=[];position=0;seen=[]
    def hook(layer):
        def capture_gate(module,args,logits):
            probs=torch.softmax(logits,dim=-1,dtype=torch.float32)
            weight,selected=torch.topk(probs,8,dim=-1);weight=(weight/weight.sum(-1,keepdim=True)).float()
            n=logits.shape[0]//CONTEXT;assert n==CAPTURE_BATCH
            arrays['selected'][position:position+n,layer]=selected.view(n,CONTEXT,8).cpu().numpy()
            arrays['weights'][position:position+n,layer]=weight.view(n,CONTEXT,8).cpu().numpy()
            arrays['gates'][position:position+n,layer]=probs.view(n,CONTEXT,128).mean(1).cpu().numpy()
            seen.append(layer)
        return capture_gate
    hooks=[b.mlp.gate.register_forward_hook(hook(i)) for i,b in enumerate(model.model.layers)]
    with torch.inference_mode():
        for position in range(0,512,CAPTURE_BATCH):
            if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
            memory();seen.clear()
            ids=torch.tensor([r['input_ids'][-CONTEXT:] for r in rows[position:position+CAPTURE_BATCH]],device='cuda')
            assert ids.shape==(CAPTURE_BATCH,CONTEXT);mask=torch.ones_like(ids)
            result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
            assert seen==list(range(48)) and torch.isfinite(result.logits).all()
            generated.extend(result.logits[:,-1].argmax(-1).cpu().tolist())
            del result,ids,mask
            if position%32==0:
                for x in arrays.values():x.flush()
                write(out/'progress.json',dict(status='CAPTURING',completed=position+CAPTURE_BATCH,total=512,**memory()))
    for h in hooks:h.remove()
    for x in arrays.values():x.flush()
    np.save(out/'first_tokens.npy',np.asarray(generated,np.int64))
    write(out/'receipt.json',dict(status='PASS',physical_gpu=a.gpu,request_ids=[r['request_id'] for r in rows],
        context=CONTEXT,capture_batch=CAPTURE_BATCH,source_request_manifest_sha256=sha(OLD_ROOT/'requests.json'),
        files={f.name:sha(f) for f in out.glob('*.npy')},**memory()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
