"""CPU-only bit/order/shape parity of packed indices on saved physical events."""
import ast,gzip,json,pickle
from pathlib import Path
import torch
from env_offload_tensors import pack_layouts
root=Path(__file__).resolve().parents[1];path=Path('/home/hwlee/mgo-results/env_e2e_tpot_offload_20261003/P_BR_env1_PLAN_3')
node=next(n for n in ast.parse((path/'worker_source.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='device_layout')
class CPU(ast.NodeTransformer):
 def visit_Constant(self,node):
  return ast.copy_location(ast.Constant(value='cpu'),node) if node.value=='cuda' else node
node=CPU().visit(node);scope={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<reference tensor layout>','exec'),scope)
with gzip.open(path/'rank0.pkl.gz','rb') as f:events=pickle.load(f)
chosen=[events[i] for i in [0,47,48,96,4096,12335]];actual=pack_layouts(chosen,'cpu');elements=0
for e,a in zip(chosen,actual):
 ref=scope['device_layout'](e)
 for key in ['send_idx','send_eids','return_order','targets','selected']:
  assert torch.equal(a[key],ref[key]) and a[key].shape==ref[key].shape and a[key].dtype==ref[key].dtype and a[key].is_contiguous();elements+=a[key].numel()
 for x,y in zip(a['groups'],ref['groups']):assert x[0]==y[0] and x[3]==y[3] and torch.equal(x[1],y[1]) and torch.equal(x[2],y[2]);elements+=x[1].numel()+x[2].numel()
 for x,y in zip(a['combine'],ref['combine']):assert torch.equal(x[0],y[0]) and torch.equal(x[1],y[1]);elements+=x[0].numel()+x[1].numel()
 assert a['fetches']==ref['fetches'] and a['send_counts']==ref['send_counts'] and a['recv_counts']==ref['recv_counts'] and a['return_counts']==ref['return_counts'] and a['return_recv_counts']==ref['return_recv_counts']
assert len({e['selected'].untyped_storage().data_ptr() for e in actual})==1
result=dict(status='PASS',physical_events=len(chosen),integer_elements_compared=elements,shared_index_storage=True,device='CPU validation; GPU warmup gate remains mandatory')
(root/'experiments/env_e2e_tpot_offload_20261003/tensor_layout_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
