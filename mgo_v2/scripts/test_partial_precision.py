"""CPU numerical diagnostic: rank partials versus exact BF16-contribution sum."""
import json
import torch

def check():
 torch.set_num_threads(1);torch.manual_seed(42);rows=[]
 for scale in ('normal','wide'):
  values=torch.randn((2048,8),dtype=torch.bfloat16)
  if scale=='wide':values=(values*torch.logspace(-6,6,8)).bfloat16()
  owner=torch.randint(0,8,values.shape)
  exact=values.double().sum(1).bfloat16();legacy=torch.zeros(2048,dtype=torch.bfloat16)
  for k in range(8):legacy+=values[:,k]
  for dtype in (torch.bfloat16,torch.float32,torch.float64):
   partial=torch.zeros((2048,8),dtype=dtype)
   for k in range(8):partial.scatter_add_(1,owner[:,k:k+1],values[:,k:k+1].to(dtype))
   result=torch.zeros(2048,dtype=dtype)
   for rank in range(8):result+=partial[:,rank]
   result=result.bfloat16();rows.append(dict(scale=scale,accumulation=str(dtype),diff_vs_legacy=int((result!=legacy).sum()),diff_vs_exact=int((result!=exact).sum()),max_abs_error_vs_exact=float((result.float()-exact.float()).abs().max())))
  assert rows[-1]['diff_vs_exact']==0
 print(json.dumps(dict(status='PASS',scope='small CPU numerical diagnostic, not model accuracy or performance',rows=rows),indent=2))
if __name__=='__main__':check()
