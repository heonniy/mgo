"""One-cell validation of balanced 3/3/3/3 static GPU-expert placement."""
import json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parents[1]
PACK=P/'experiments/main_table_global_workload_20261006/expanded_matrix'
ROOT=Path('/home/hwlee/mgo-results/headline_r4_20261007')
TOOLS=Path('/home/hwlee/mgo-tools/headline-r4')
PY='/home/hwlee/sub-moe/phase01/.venv/bin/python'
BASE=str(TOOLS/'base-env/bin/python')
CELL='R4_C30_B32_L512_O64'
LABEL='llama_balanced3_R4_C30_B32_L512_O64_v1'
REPEATS=2

def mean(xs):return sum(xs)/len(xs)
def main():
 subprocess.run([PY,str(P/'scripts/build_headline_llama_sync.py')],check=True)
 out=ROOT/LABEL
 if not (out/'status.json').exists():
  cmd=[PY,str(P/'scripts/run_headline_job.py'),
       '--label',LABEL,'--system','llama.cpp-sync-balanced3',
       '--worker','headline_llama_sync_worker.py','--cell',CELL,
       '--python',BASE,'--ranks','1','--repeats',str(REPEATS),'--timeout','28800',
       '--workloads',str(PACK/'WORKLOADS.json'),
       '--llama-threads','32','--llama-cuda-graphs','off','--llama-graph-reuse','off',
       '--llama-expert-placement','balanced3']
  subprocess.run(cmd,check=True)
 status=json.loads((out/'status.json').read_text());assert status['status']=='PASS'
 cfg=json.loads((out/'config.json').read_text())
 placement=json.loads((out/'placement_audit.json').read_text())
 assert cfg['cpu_threads']==32 and cfg['cpu_batch_threads']==32
 assert cfg['cuda_graphs_runtime']=='off' and cfg['llama_graph_reuse'] is False
 assert cfg['op_offload'] is False and cfg['expert_placement']=='balanced3'
 assert placement['gpu_expert_layer_ids']==[2,6,10,14,18,22,26,30,34,38,42,46]
 assert sorted(placement['gpu_expert_layers_by_device'].values())==[3,3,3,3]
 rows=[json.loads((out/f'repeat{r}.json').read_text()) for r in range(1,REPEATS+1)]
 summary=dict(
  status='PASS',cell=CELL,local_batch=32,global_requests=128,input_tokens=512,output_tokens=64,
  cache_percent=30,cpu_threads=32,cuda_graphs='off',graph_reuse='off',op_offload=False,
  expert_placement='balanced3',gpu_expert_layers=12,cpu_expert_layers=36,
  gpu_expert_layer_ids=placement['gpu_expert_layer_ids'],
  gpu_expert_layers_by_device=placement['gpu_expert_layers_by_device'],
  static_gpu_expert_bytes=placement['gpu_expert_bytes'],
  c30_budget_bytes=cfg['expert_budget_bytes'],
  unused_c30_bytes=cfg['expert_budget_bytes']-placement['gpu_expert_bytes'],
  repeats=REPEATS,
  TTFT=[r['TTFT'] for r in rows],TPOT=[r['TPOT'] for r in rows],E2E=[r['E2E'] for r in rows],
  TTFT_mean=mean([r['TTFT'] for r in rows]),
  TPOT_mean=mean([r['TPOT'] for r in rows]),
  E2E_mean=mean([r['E2E'] for r in rows]))
 path=PACK/'LLAMA_BALANCED3_B32_L512_SUMMARY.json'
 path.write_text(json.dumps(summary,indent=2)+'\n')
 print(path)
if __name__=='__main__':main()
