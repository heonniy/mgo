"""Drive native synchronous llama batch using exact frozen token manifests."""
import argparse,hashlib,json,os,subprocess,re
from pathlib import Path
P=Path(__file__).resolve().parents[1]
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main(a):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1,4,5'
 spec=next(s for s in json.loads(Path(os.environ['MGO_HEADLINE_WORKLOADS']).read_text())['cells'] if s['cell']==a.cell)
 binary=TOOLS/'llama.cpp/build/bin/headline-llama-sync';build=json.loads((P/'experiments/main_table_global_workload_20261006/expanded_matrix/LLAMA_BUILD.json').read_text())
 assert hashlib.sha256(binary.read_bytes()).hexdigest()==build['binary_sha256']
 layers=spec['expert_slots']//128;resident=layers*128*9*2**20;assert resident<=spec['expert_budget_bytes']
 cmd=[str(binary),str(TOOLS/'Qwen3-30B-A3B-Instruct-2507-BF16.gguf'),spec['warmup']['path'],spec['target']['path'],str(a.output),str(layers),str(1 if a.smoke else a.repeats),str(int(a.smoke))]
 write(a.output/'config.json',dict(command=cmd,build=build,expert_budget_bytes=spec['expert_budget_bytes'],expert_resident_bytes=resident,gpu_expert_layers=layers,cpu_expert_layers=48-layers,synchronous_batch=True,op_offload=False))
 subprocess.run(cmd,check=True)
 lines=[s for s in (a.output/'run.log').read_text().splitlines() if 'KV buffer size' in s]
 assert len(lines)==4 and all(re.search(r'CUDA[0-3] KV buffer',s) for s in lines),lines
 write(a.output/'kv_placement.json',dict(status='PASS',buffers=lines))
 for r in range((1 if a.smoke else a.repeats)+1):
  result=json.loads((a.output/f'repeat{r}.json').read_text());n=2 if a.smoke else 64;count=4 if a.smoke else spec['global_requests']
  assert result['synchronous_batch'] and result['finite_logits'] and len(result['tokens'])==count
  assert all(len(t)==n for t in result['tokens']) and len(result['token_ready_ns'])==n
  assert all(x<y for x,y in zip(result['token_ready_ns'],result['token_ready_ns'][1:]))
 write(a.output/'result.json',dict(status='PASS',system='llama.cpp-sync',cell=a.cell,smoke=a.smoke,primary_repeats=1 if a.smoke else a.repeats,synchronous_batch=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--repeats',type=int,choices=range(1,6),default=5);main(p.parse_args())
