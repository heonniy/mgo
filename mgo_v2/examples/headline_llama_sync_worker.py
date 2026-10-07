"""Drive native synchronous llama batch with explicit, audited CPU budget."""
import argparse,hashlib,json,os,subprocess,re
from pathlib import Path
P=Path(__file__).resolve().parents[1]
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
TOPOLOGY=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json')
GPUS=[0,1,4,5]
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def balanced_affinity(threads):
 assert threads in (16,32,64) and threads%4==0
 fixed=json.loads(TOPOLOGY.read_text())['fixed_affinity'];per=threads//4;chosen=[];pools={}
 for gpu in GPUS:
  pool=list(fixed[str(gpu)]);assert len(pool)>=per,(gpu,len(pool),per)
  pools[str(gpu)]=pool;chosen+=pool[:per]
 assert len(chosen)==threads and len(set(chosen))==threads
 return chosen,pools
def main(a):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,4,5'
 spec=next(s for s in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if s['cell']==a.cell)
 binary=TOOLS/'llama.cpp/build/bin/headline-llama-sync'
 build_path=P/'experiments/main_table_global_workload_20261006/expanded_matrix/LLAMA_BUILD.json'
 build=json.loads(build_path.read_text())
 source=P/'examples/headline_llama_sync.cpp'
 assert hashlib.sha256(source.read_bytes()).hexdigest()==build['source_sha256'],'llama sync source changed: rebuild and refresh LLAMA_BUILD.json'
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256'],'llama sync binary does not match build receipt'
 assert build.get('cmake_flags',{}).get('GGML_CUDA_GRAPHS:BOOL')=='OFF','headline llama baseline requires GGML_CUDA_GRAPHS=OFF; rebuild first'
 assert build.get('cuda_graphs') is False and build.get('baseline_policy_eligible') is True,'stale llama build receipt'
 layers=spec['expert_slots']//128
 resident=layers*128*9*2**20
 assert resident<=spec['expert_budget_bytes']
 affinity,pools=balanced_affinity(a.threads)
 os.sched_setaffinity(0,set(affinity))
 assert set(os.sched_getaffinity(0))==set(affinity)
 cmd=[str(binary),str(TOOLS/'Qwen3-30B-A3B-Instruct-2507-BF16.gguf'),spec['warmup']['path'],spec['target']['path'],str(a.output),str(layers),str(1 if a.smoke else a.repeats),str(int(a.smoke)),str(a.threads)]
 env=dict(os.environ)
 env['OMP_THREAD_LIMIT']=str(a.threads)
 env['LLAMA_GRAPH_REUSE_DISABLE']='0'  # keep ordinary llama graph reuse enabled; CUDA Graphs are build-time OFF
 write(a.output/'config.json',dict(
  command=cmd,build=build,expert_budget_bytes=spec['expert_budget_bytes'],expert_resident_bytes=resident,
  gpu_expert_layers=layers,cpu_expert_layers=48-layers,synchronous_batch=True,op_offload=False,
  cpu_threads=a.threads,cpu_batch_threads=a.threads,cpu_affinity=affinity,
  affinity_policy='equal slice from the existing GPU-local fixed-affinity pools; child and llama threadpool inherit this mask',
  source_affinity_pools=pools,omp_thread_limit=a.threads,
  cuda_graphs=False,llama_graph_reuse=True))
 subprocess.run(cmd,check=True,env=env)

 placement=json.loads((a.output/'placement_audit.json').read_text())
 assert placement['status']=='PASS'
 assert placement['cpu_expert_layers']==48-layers and placement['gpu_expert_layers']==layers
 assert placement['cpu_expert_tensor_count']==(48-layers)*3
 assert placement['gpu_expert_tensor_count']==layers*3
 assert placement['expected_total_expert_tensor_count']==144
 assert placement['gpu_expert_bytes']==resident and placement['op_offload'] is False

 lines=[s for s in (a.output/'run.log').read_text().splitlines() if 'KV buffer size' in s]
 assert len(lines)==4 and all(re.search(r'CUDA[0-3] KV buffer',s) for s in lines),lines
 write(a.output/'kv_placement.json',dict(status='PASS',buffers=lines))

 for r in range((1 if a.smoke else a.repeats)+1):
  result=json.loads((a.output/f'repeat{r}.json').read_text())
  n=2 if a.smoke else 64;count=4 if a.smoke else spec['global_requests']
  assert result['synchronous_batch'] and result['finite_logits'] and len(result['tokens'])==count
  assert result['cpu_threads']==a.threads==result['cpu_batch_threads']
  assert result['op_offload'] is False and result['offload_kqv'] is True
  assert all(len(t)==n for t in result['tokens']) and len(result['token_ready_ns'])==n
  assert len(result['decode_step_seconds'])==n-1
  assert all(x<y for x,y in zip(result['token_ready_ns'],result['token_ready_ns'][1:]))
  recomputed=sum(result['decode_step_seconds'])/(n-1)
  assert abs(recomputed-result['TPOT'])<1e-6,(recomputed,result['TPOT'])
  expected_calls=(count*result['input_tokens']+result['n_batch']-1)//result['n_batch']
  assert result['prefill_decode_calls']==expected_calls
  assert result['decode_calls']==n-1 and result['decode_tokens_per_call']==count
 write(a.output/'result.json',dict(
  status='PASS',system='llama.cpp-sync',cell=a.cell,smoke=a.smoke,
  primary_repeats=1 if a.smoke else a.repeats,synchronous_batch=True,
  cpu_threads=a.threads,cpu_batch_threads=a.threads,cpu_affinity=affinity,
  placement_audit='PASS',metric_recompute='PASS',cuda_graphs=False,llama_graph_reuse=True))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=range(1,6),default=5)
 p.add_argument('--threads',type=int,choices=(16,32,64),required=True)
 main(p.parse_args())
