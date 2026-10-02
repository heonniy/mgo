#!/usr/bin/env python3
"""Idle GPU load using a resident Qwen MoE model and repeated real forwards."""
import json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path('/home/hwlee/mgo-results/model_inference_load_20261003')
MODEL='/home/hwlee/model/Qwen1.5-MoE-A2.7B-Chat'
physical=int(os.environ['CUDA_VISIBLE_DEVICES']);stopping=False

def stop(*_):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
uuid=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()

def guard():
 if stopping or (ROOT/'STOP').exists():return 'requested stop'
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
 if any(g.strip()==uuid and int(pid)!=os.getpid() for g,pid in (line.split(',') for line in apps.splitlines())):return 'other compute process appeared'
 row=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=temperature.gpu,memory.free','--format=csv,noheader,nounits'],text=True)
 temp,free=map(int,row.split(','))
 if temp>=85:return f'temperature {temp} C'
 if free<8192:return f'GPU free {free} MiB'
 mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
 if int(mem['MemAvailable'].split()[0])<128*1024**2:return 'host memory guard'
 return None

def main():
 reason=guard()
 if reason:print(reason,flush=True);return
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 torch.set_num_threads(1);torch.cuda.set_device(0)
 torch.cuda.set_per_process_memory_fraction(.65,0)
 print(json.dumps(dict(event='LOAD',gpu=physical,pid=os.getpid(),model=MODEL)),flush=True)
 tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side='left')
 if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
 model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,device_map={'':'cuda:0'},low_cpu_mem_usage=True,attn_implementation='sdpa').eval()
 questions=json.loads(Path('/home/hwlee/mgo-results/runtime_validation_20261001/screen_workload.json').read_text())
 texts=[tokenizer.apply_chat_template([{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True) for r in questions[:8]]
 inputs={k:v.cuda() for k,v in tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt').items()}
 iteration=0;last_guard=0;last_report=0;reason=None
 with torch.inference_mode():
  while not stopping:
   now=time.monotonic()
   if now-last_guard>=5:
    reason=guard()
    if reason:break
    last_guard=now
   out=model(**inputs,use_cache=False);last_tokens=out.logits[:,-1].argmax(-1);torch.cuda.synchronize();del out
   iteration+=1
   if now-last_report>=30:
    state=dict(status='RUNNING',gpu=physical,pid=os.getpid(),model=MODEL,batch=8,input_shape=list(inputs['input_ids'].shape),iterations=iteration,allocated_mib=torch.cuda.memory_allocated()/2**20,generated_last_token_ids=last_tokens.cpu().tolist(),unix=time.time())
    target=ROOT/f'gpu{physical}.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(target)
    print(json.dumps(state),flush=True);last_report=now
 print(json.dumps(dict(event='STOP',gpu=physical,reason=reason or 'signal',iterations=iteration)),flush=True)
if __name__=='__main__':main()
