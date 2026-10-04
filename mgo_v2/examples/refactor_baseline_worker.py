"""Short full-model baseline gate for BR/CA/LA; never reports speed claims."""
from la_physical_worker import *
class PrefixRuntime(LiveRuntime):
 def reset(self):
  self.policy_kind={'BR':0,'CA':1,'LA':4}[self.args.policy]
  super().reset()

def main(a):
 rank=int(os.environ['RANK']);a.output.mkdir(parents=True,exist_ok=True)
 cpus=json.loads(Path('/home/hwlee/mgo-results/timing_stability_numa_20261004/topology.json').read_text())['fixed_affinity'][str(rank)]
 for task in Path('/proc/self/task').iterdir():os.sched_setaffinity(int(task.name),cpus)
 torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.85);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 dist.init_process_group('nccl',device_id=torch.device('cuda:0'))
 source=json.loads((a.inputs/'receipt.json').read_text());a.seed=source['winner']['placement_seed'];a.capacities=[3686//8+(r<3686%8) for r in range(8)];a.comm_mode='current';a.phase='COUNTERS'
 records=json.loads((a.inputs/'requests.json').read_text())['ranks'][rank];batch=len(records)
 model,backing,experts=load_model();length=max(len(r['input_ids']) for r in records);pad=model.generation_config.pad_token_id
 ids=torch.tensor([[pad]*(length-len(r['input_ids']))+r['input_ids'] for r in records],device='cuda');mask=torch.tensor([[0]*(length-len(r['input_ids']))+[1]*len(r['input_ids']) for r in records],device='cuda')
 teacher=torch.tensor(np.load(a.inputs/'teacher.npy')[rank*batch:(rank+1)*batch],device='cuda');rows=[]
 for name in ['BR','CA','LA']:
  a.policy=name;rt=PrefixRuntime(a,model,backing,experts);expected=None
  for repeat in range(2):
   rt.reset();row,tokens=generate(model,rt,ids,mask,teacher,8)
   assert row['state_hash']==rt.proof['rank_state_hashes'][rank]
   assert rt.actual_h2d_bytes==rt.proof['H2D_bytes'][rank] and not rt.mismatch.item()
   if expected is not None:assert np.array_equal(expected,tokens)
   expected=tokens
   rows.append(dict(policy=name,repeat=repeat,argmax_hash=row['argmax_hash'],state_hash=row['state_hash'],H2D_bytes=rt.actual_h2d_bytes))
  # Detach closures before reclaiming the arena for the next policy.
  for block in model.model.layers:block.mlp.forward=lambda *args: None
  del rt;gc.collect();torch.cuda.empty_cache()
 write(a.output/f'rank{rank}.json',dict(status='PASS',rank=rank,horizon=8,rows=rows,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()));dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
