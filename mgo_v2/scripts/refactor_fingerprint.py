"""Common-stack identity, recorded outside timing before each paired group."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parents[1]
INPUTS=('selected.npy','weights.npy','offsets.npy','prefill_origins.npy','decode_origins.npy','gates.npy','teacher.npy','requests.json')

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def capture(inputs,env):
 paths=list((P/'mgo_v2').glob('*.py'))
 paths += [P/'examples'/name for name in ('env_offload_worker.py','fetch_relaxed_worker.py','la_physical_worker.py','refactor_baseline_worker.py','refactor_measure_worker.py','refactor_paired_worker.py')]
 paths += [P/'scripts'/name for name in ('env_offload_policy.py','env_offload_layout.py','env_offload_tensors.py','br_carep_cpu.py','la_placement.py','old_ca_fanout_policy.py','adaptive_timing.py','refactor_thread_usage.py','refactor_thread_affinity.py')]
 code={str(p.relative_to(P)):sha(p) for p in sorted(paths)}
 inputs_hashes={name:sha(inputs/name) for name in INPUTS}
 settings={k:v for k,v in env.items() if k.startswith(('NCCL_','MGO_V2_','MOE_EP_')) or k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS','CUBLAS_WORKSPACE_CONFIG')}
 affinity=Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json')
 predictor=Path('/home/hwlee/mgo-results/decode_prefetch_runtime_refactoring_20261004/predictor/transition.npy')
 return dict(code=code,inputs=inputs_hashes,environment=settings,affinity_sha256=sha(affinity),predictor_sha256=sha(predictor),input_receipt_sha256=sha(inputs/'receipt.json'),precision='bf16',scope='Frozen input bytes, runtime/kernel/measurement sources, predictor, CPU affinity and selected nonsecret runtime environment. Model weight files are not rehashed here.')

def assert_equivalent(identities):
 assert identities
 first=next(iter(identities.values()))
 for identity in identities.values():
  for key in ('code','environment','affinity_sha256','predictor_sha256','precision'):
   assert identity[key]==first[key],('common-stack mismatch',key)
 batches={batch for arm,batch in identities}
 for batch in batches:
  rows=[identity for (arm,b),identity in identities.items() if b==batch]
  for row in rows[1:]:
   for key in ('inputs','input_receipt_sha256'):assert row[key]==rows[0][key],('frozen-input mismatch',batch,key)
