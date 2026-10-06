"""Static owner baseline using unchanged strict decode32 timing machinery."""
from strict_decode32_worker import *
from strict_static_mod_policy import StaticPolicy
import strict_decode32_worker as original

POLICY_NAMES=('STATIC_MOD',)

class GenerationRuntime(original.GenerationRuntime):
 def reset(self):
  saved=self.args.policy
  self.args.policy='BR'
  try:super().reset()
  finally:self.args.policy=saved
  self.policy_kind=8
  self.policy=StaticPolicy(self.args.capacities,self.similarity,False,8,self.args.seed)

 def plan_event(self,layer,selected,weights,probs):
  assert not self.capturing
  e=super().plan_event(layer,selected,weights,probs)
  if self.args.phase!='MEASURE':assert all((key%128)%self.world==self.rank for key,slot,victim,rep in e['fetches'])
  return e

def validation(rt,row,rank):
 v=original.validation(rt,row,rank)
 ref=json.loads((Path(rt.args.inputs)/f'reference_BR_warm_rank{rank}.json').read_text())
 v['tokens_match_reference_BR']=row['argmax_hash']==ref['argmax_hash']
 if not v['tokens_match_reference_BR']:v['status']='FAIL'
 return v

def main(a):
 rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);physical=list(map(int,os.environ['MGO_V2_PHYSICAL_GPUS'].split(',')));assert len(physical)==world
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(physical[rank])]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 spec=json.loads(a.specs.read_text())[0];reference=json.loads(Path(spec['reference_result']).read_text());source=Path(spec['inputs']);r=json.loads((source/'receipt.json').read_text());a.inputs=source;a.capacities=r['capacities'];a.seed=r['placement_seed'];a.comm_mode='current';a.staging_cpu_team=cpus[1:3];a.output.mkdir(exist_ok=True,parents=True)
 model,backing,experts=load_model();records=json.loads((source/'requests.json').read_text())['ranks'][rank];ids=torch.tensor([x['input_ids'] for x in records],device='cuda');batch=len(ids)
 def setup(policy,phase,capture=False):
  a.policy=policy;a.phase=phase;a.capture=capture;rt=GenerationRuntime(a,model,backing,experts)
  from refactor_thread_affinity import configure
  configure(cpus,rt.h2d.thread.native_id,True,rt.h2d.cpu_team_receipt);return rt
 frozen=source
 assert (frozen/'receipt.json').exists(), 'static baseline must reuse frozen input; no recapture'
 a.inputs=frozen;teacher=torch.tensor(np.load(frozen/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda');warm={};samples={p:[] for p in POLICY_NAMES}
 for policy in POLICY_NAMES:
  write(a.output/f'phase_rank{rank}.json',dict(stage='WARM_CORRECTNESS',policy=policy));rt=setup(policy,'COUNTERS');row,tokens=generation(model,rt,ids,teacher);v=validation(rt,row,rank);assert v['status']=='PASS';warm[policy]=row['argmax_hash'];write(a.output/f'{policy}_warm_rank{rank}.json',dict(**row,validation=v));rt.close();del rt;gc.collect();dist.barrier()
 flags=[None]*world;dist.all_gather_object(flags,len(set(warm.values()))==1);assert all(flags),'cross-policy frozen32 token mismatch'
 order=list(POLICY_NAMES) if spec['order']==0 else list(reversed(POLICY_NAMES))
 for repeat in [1,2,3]:
  if repeat==3:
   delta=max(abs(samples[p][0][m]-samples[p][1][m])/statistics.mean([samples[p][0][m],samples[p][1][m]]) for p in POLICY_NAMES for m in METRICS)
   if delta<=.02 or delta>.05:break
  for policy in (order if repeat%2 else list(reversed(order))):
   rt=setup(policy,'MEASURE');gc.collect();torch.cuda.synchronize();dist.barrier();label=f'{policy}_r{repeat}'
   from torch._dynamo.utils import counters
   before=dict(counters['stats']);write(a.output/f'{label}_ready_rank{rank}.json',dict(rank=rank));dist.barrier()
   if rank==0:write(a.output/'boundary.json',dict(key=label))
   deadline=time.monotonic()+600
   while not (a.output/f'{label}_GO').exists():
    if time.monotonic()>deadline:raise TimeoutError('release boundary')
    time.sleep(.2)
   with torch._dynamo.config.patch(error_on_recompile=True):row,tokens=generation(model,rt,ids,teacher)
   v=validation(rt,row,rank);v.update(no_compile=before==dict(counters['stats']),tokens_match_warm=row['argmax_hash']==warm[policy]);write(a.output/f'{label}_validation_rank{rank}.json',v)
   assert v['status']=='PASS' and v['no_compile'] and v['tokens_match_warm']
   write(a.output/f'{label}_measure_rank{rank}.json',dict(status='PASS',**row,validation=v));rt.close();del rt;gc.collect();dist.barrier()
   rr=[json.loads((a.output/f'{label}_measure_rank{k}.json').read_text()) for k in range(world)];samples[policy].append({m:max(x[m] for x in rr) for m in METRICS})
 if rank==0:
  estimates={p:{m:statistics.median(x[m] for x in samples[p]) for m in METRICS} for p in POLICY_NAMES}
  differences={p:{m:abs(samples[p][0][m]-samples[p][1][m])/statistics.mean([samples[p][0][m],samples[p][1][m]]) for m in METRICS} for p in POLICY_NAMES}
  unstable={p:{m:(max(x[m] for x in samples[p])-min(x[m] for x in samples[p]))/statistics.mean(x[m] for x in samples[p])>.05 for m in METRICS} for p in POLICY_NAMES}
  write(a.output/'result.json',dict(status='PASS',spec=spec,samples=samples,estimates=estimates,relative_difference=differences,unstable=unstable,gains={p:{m:1-estimates[p][m]/reference['estimates']['BR'][m] for m in METRICS} for p in POLICY_NAMES if p!='BR'},reference_BR_estimates=reference['estimates']['BR'],reference_result=spec['reference_result'],decode_steps=32,diagnostic_passes=0,scope='STATIC_MOD frozen BR-generated tokens/routes; same static owner in prefill/decode. Sequential addition: earlier BR timing is a historical reference, not an interleaved pair; session drift is possible. First output from prefill plus32 subsequent decode outputs. E2E/TPOT are wall time; per-metric maximum across ranks.'))
 dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--specs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
