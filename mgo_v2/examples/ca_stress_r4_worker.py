"""Owner-frozen R4 stress cell on one explicitly selected GPU set, using the validated physical kernels and timing loop."""
from env_offload_worker import *
from timing_stability_residency_worker import generate
VARIANT=os.environ['MGO_R4_GPU_SET'];assert VARIANT in ('0123','0146')
RUN=Path('/home/hwlee/mgo-results/ca_stress_physical_validation_20261004')/('R4_'+VARIANT)
import env_offload_worker as base
class Seed73Policy(base.Policy):
 def __init__(self,capacities,similarity,substitution,policy,seed=73):
  assert seed==73
  super().__init__(capacities,similarity,substitution,policy,seed=73)
base.Policy=Seed73Policy

def main(a):
 rank=int(os.environ['RANK']);topology=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())
 cpus=topology['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'));assert dist.get_world_size()==4
 a.cell='STRESS';a.dataset='ShareGPT';a.batch=8;a.substitution=False;a.horizon=256;a.residency='NORMAL'
 total=48*128*30//100;a.capacities=[total//4+(r<total%4) for r in range(4)]
 manifest=json.loads((RUN/'requests.json').read_text());records=manifest['ranks'][rank];assert len(records)==8
 model,backing,experts=load_model();rt=Runtime(a,model,experts,a.capacities[rank]);rt.cpu_backing=backing
 length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 if a.phase in ('PLAN','COUNTERS'):
  receipt,tokens=generate(model,rt,ids,mask)
  if a.phase=='PLAN':
   with gzip.open(a.output/f'rank{rank}.pkl.gz','wb') as f:pickle.dump(rt.events,f,protocol=4)
   receipt['action_hash']=digest(rt.events)
  else:
   expected=json.loads((a.plan/f'rank{rank}.json').read_text());assert all(receipt[k]==expected[k] for k in ['token_hash','state_hash'])
   receipt.update(actual_H2D_bytes=rt.actual_h2d_bytes,actual_peer_bytes=rt.actual_peer_bytes,logical_counters=np.asarray(rt.metrics).sum(0).tolist())
  np.save(a.output/f'rank{rank}_metrics.npy',np.asarray(rt.metrics));np.save(a.output/f'rank{rank}_tokens.npy',tokens)
 else:
  expected=json.loads((a.plan/f'rank{rank}.json').read_text());warm,tokens=generate(model,rt,ids,mask)
  assert all(warm[k]==expected[k] for k in ['token_hash','state_hash'])
  if a.phase=='COMPILE':receipt=warm
  else:
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);rt.reset();torch.manual_seed(42);torch.cuda.synchronize();dist.barrier()
   write(a.output/f'ready{rank}.json',dict(rank=rank,warmup_hashes_valid=True,affinity=cpus,unix=time.time()))
   deadline=time.monotonic()+600
   while not (a.output/'GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('boundary coordinator not ready')
    time.sleep(.1)
   with torch._dynamo.config.patch(error_on_recompile=True):receipt,tokens=generate(model,rt,ids,mask)
   assert all(receipt[k]==expected[k] for k in ['token_hash','state_hash'])
   assert dict(counters['stats'])==before,'compilation in MEASURE'
   receipt.update(no_compile_in_measure=True,compiler_stats=before)
 receipt.update(status='PASS',phase=a.phase,policy=a.policy,environment=a.environment,rank=rank,physical_gpu=int(VARIANT[rank]),BR_seed=73,affinity=cpus,residency='NORMAL',boot=BOOT,cache_capacity=a.capacities[rank],max_resident_copies=int(np.count_nonzero(rt.keys>=0)),request_ids=[r['request_id'] for r in records],request_manifest_sha256=hashlib.sha256((RUN/'requests.json').read_bytes()).hexdigest(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
 if a.phase!='MEASURE':
  for k in ['E2E_wall','decode_wall','TPOT']:receipt.pop(k,None)
 if a.phase!='PLAN':receipt.update(route_hash=rt.plan_proof['route_hash'],action_hash=rt.plan_proof['action_hash'],schedule_file_sha256=rt.plan_proof['schedule_file_sha256'],route_validation='all frozen selected experts compared on device')
 write(a.output/f'rank{rank}.json',receipt);dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--policy',choices=['BR','CA'],required=True);p.add_argument('--phase',choices=['PLAN','COMPILE','MEASURE','COUNTERS'],required=True);p.add_argument('--environment',choices=['env1','env2'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plan',type=Path);main(p.parse_args())
