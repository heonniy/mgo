"""Native llama.cpp layer split with exact-token readiness timestamps."""
import argparse,concurrent.futures,json,os,re,socket,subprocess,threading,time,urllib.request
from pathlib import Path
import psutil
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
def write(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2));q.replace(p)
def request(url,payload=None,timeout=7200):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)
def main(a):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,4,5'
 spec=next(x for x in json.loads((ROOT/'WORKLOADS.json').read_text())['cells'] if x['cell']==a.cell)
 count=4 if a.smoke else spec['local_batch']*4;length=32 if a.smoke else spec['input_tokens'];n=2 if a.smoke else 64
 sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close();url=f'http://127.0.0.1:{port}'
 cmd=[str(TOOLS/'llama.cpp/build/bin/llama-server'),'-m',str(TOOLS/'Qwen3-30B-A3B-Instruct-2507-BF16.gguf'),'--host','127.0.0.1','--port',str(port),'--split-mode','layer','--n-gpu-layers','999','--n-cpu-moe','34','--parallel',str(count),'--ctx-size',str(count*(length+n+8)),'--cache-ram','0','--cache-reuse','0','--threads','32','--threads-batch','64','--threads-http','16','--batch-size','2048','--ubatch-size','512','--no-context-shift','--no-op-offload']
 if a.smoke:cmd+=['--log-verbosity','4']
 write(a.output/'config.json',dict(command=cmd,global_expert_slots=14*128,expert_bytes=14*128*9*2**20,expert_budget_bytes=17392730112,cache_mode='static layer weights; no prompt reuse; host expert op offload disabled',eos_semantics='continue after EOS without suppressing its logit'))
 with (a.output/'server.log').open('w') as log:
  server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MGO_HEADLINE_CONTINUE_EOG="1"))
  try:
   deadline=time.monotonic()+900
   while True:
    if server.poll() is not None:raise RuntimeError('llama-server exited during startup')
    try:
     if request(url+'/health',timeout=1).get('status')=='ok':break
    except Exception:pass
    if time.monotonic()>deadline:raise TimeoutError('server startup')
    time.sleep(1)
   if a.smoke:
    lines=(a.output/'server.log').read_text().splitlines()
    kv_lines=[line for line in lines if 'KV buffer size =' in line]
    assert kv_lines and all(re.search(r'CUDA[0-3] KV buffer size',line) for line in kv_lines),kv_lines
    write(a.output/'kv_placement.json',dict(status='PASS',evidence='actual smoke startup allocations',buffers=kv_lines,scope='smoke shape; primary placement uses the same all-layer GPU assignment'))
   for repeat in range((1 if a.smoke else 3)+1):
    phase='warmup' if repeat==0 else 'target';rows=json.loads(Path(spec[phase]['path']).read_text())['requests'][:count]
    write(a.output/'phase.json',dict(system='llama.cpp-layer',phase=phase,repeat=repeat,cell=a.cell,smoke=a.smoke))
    barrier=threading.Barrier(count+1)
    def run(row):
     payload=dict(prompt=row['input_ids'][-length:],n_predict=n,temperature=0,ignore_eos=False,cache_prompt=False,return_tokens=True,stream=False,seed=42)
     barrier.wait(timeout=30);result=request(url+'/completion',payload)
     assert result['tokens_evaluated']==length and result['tokens_predicted']==n and not result['truncated'],result
     assert len(result['tokens'])==len(result['token_ready_us'])==n
     # tokens_cached includes newly populated KV; only prompt reuse count
     # in timings.prompt_n/cache_n (when present) is a reuse observation.
     result['request_id']=row['request_id'];return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
     futures=[pool.submit(run,row) for row in rows]
     deadline=time.monotonic()+30
     while barrier.n_waiting!=count:
      if time.monotonic()>deadline:raise TimeoutError('client release barrier')
      time.sleep(.001)
     start=time.perf_counter_ns();barrier.wait(timeout=30);receipts=[f.result() for f in futures]
    first=max(r['token_ready_us'][0]*1000 for r in receipts);end=max(r['token_ready_us'][-1]*1000 for r in receipts)
    assert start<=min(r['token_ready_us'][0]*1000 for r in receipts)<=first<end<=time.perf_counter_ns()
    result=dict(status='PASS',system='llama.cpp-layer',repeat=repeat,phase=phase,smoke=a.smoke,TTFT=(first-start)/1e9,TPOT=(end-first)/1e9/(n-1),E2E=(end-start)/1e9,throughput=count*n/((end-start)/1e9),global_requests=count,output_tokens=n,release_ns=start,first_ns=first,end_ns=end,host_rss_bytes=psutil.Process(server.pid).memory_info().rss,expert_resident_bytes=14*128*9*2**20,requests=receipts)
    write(a.output/f'repeat{repeat}.json',result);print(json.dumps({k:result[k] for k in ['repeat','TTFT','TPOT','E2E']}),flush=True)
   write(a.output/'result.json',dict(status='PASS',system='llama.cpp-layer',cell=a.cell,smoke=a.smoke))
  finally:
   if server.poll() is None:
    server.terminate()
    try:server.wait(timeout=20)
    except subprocess.TimeoutExpired:server.kill();server.wait()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
