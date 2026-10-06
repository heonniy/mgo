"""Small-batch resident prefill capture of the eligible fixed512 request pool."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse,time,resource
from ttft_common import *
import numpy as np
import torch
from transformers import AutoModelForCausalLM

def memory():
 free,total=torch.cuda.mem_get_info();host=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 assert free>=8*2**30 and host>=256*2**30,dict(free=free,host=host)
 return dict(gpu_free_bytes=free,host_available_bytes=host,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def main(a):
 assert a.gpu in GPUS and os.environ['CUDA_VISIBLE_DEVICES']==str(a.gpu)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 rows=json.loads((ROOT/'requests.json').read_text())['requests'];start=GPUS.index(a.gpu)*512;rows=rows[start:start+512];assert len(rows)==512
 out=ROOT/'capture'/f'gpu{a.gpu}';out.mkdir(parents=True,exist_ok=False);memory()
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
 assert len(model.model.layers)==48 and model.config.num_experts==128
 arrays={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,shape,dtype in [('selected',(512,48,512,8),'uint8'),('weights',(512,48,512,8),'float32'),('gates',(512,48,128),'float32')]}
 generated=[];position=0;seen=[]
 def hook(layer):
  def capture(module,args,logits):
   probs=torch.softmax(logits,dim=-1,dtype=torch.float32);weight,selected=torch.topk(probs,8,dim=-1);weight=(weight/weight.sum(-1,keepdim=True)).to(logits.dtype).float()
   n=logits.shape[0]//512;assert n==8
   arrays['selected'][position:position+n,layer]=selected.view(n,512,8).cpu().numpy()
   arrays['weights'][position:position+n,layer]=weight.view(n,512,8).cpu().numpy()
   arrays['gates'][position:position+n,layer]=probs.view(n,512,128)[:,-128:].cpu().numpy().astype(np.float64).mean(axis=1).astype(np.float32)
   seen.append(layer)
  return capture
 hooks=[b.mlp.gate.register_forward_hook(hook(i)) for i,b in enumerate(model.model.layers)]
 with torch.inference_mode():
  for position in range(0,512,8):
   if (ROOT/'STOP').exists():raise RuntimeError('owner STOP')
   memory();seen.clear();ids=torch.tensor([r['input_ids'] for r in rows[position:position+8]],device='cuda');assert ids.shape==(8,512);mask=torch.ones_like(ids)
   result=model(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=1)
   assert seen==list(range(48)) and torch.isfinite(result.logits).all()
   generated.extend(result.logits[:,-1].argmax(-1).cpu().tolist());del result,ids,mask
   if position%32==0:
    for x in arrays.values():x.flush()
    write(out/'progress.json',dict(status='CAPTURING',completed=position+8,total=512,**memory()))
 for h in hooks:h.remove()
 for x in arrays.values():x.flush()
 np.save(out/'first_tokens.npy',np.array(generated,np.int64))
 write(out/'receipt.json',dict(status='PASS',request_ids=[r['request_id'] for r in rows],context=512,prefill_only=True,capture_batch=8,request_manifest_sha256=sha(ROOT/'requests.json'),files={f.name:sha(f) for f in out.glob('*.npy')},**memory()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);main(p.parse_args())
