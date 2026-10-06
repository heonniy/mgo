import torch,json,triton
from mgo_v2.grouped_expert import GroupedExpertExecutor
from env_offload_worker import expert_kernel
torch.manual_seed(42);torch.set_num_threads(2)
w=torch.randn((8,4718592),device='cuda',dtype=torch.bfloat16)*.02
kernel=torch.compile(expert_kernel,dynamic=True,fullgraph=True)
e=GroupedExpertExecutor(w,kernel,4096)
results=[]
for sizes in ([1],[1,2],[7,31,32,65],[1,3,9,33,65,127,256,511]):
 n=sum(sizes);e.x[:n].normal_();off=0;meta=[]
 for slot,c in enumerate(sizes):meta.append([slot,off,c,off]);off+=c
 m=torch.tensor(meta,device='cuda',dtype=torch.int32);e.math(m,sizes,n);torch.cuda.synchronize()
 diffs=[]
 for slot,off,c,_ in meta:
  ww=w[slot];ref=kernel(e.x[off:off+c],ww[:1572864].view(768,2048),ww[1572864:3145728].view(768,2048),ww[3145728:].view(2048,768))
  d=(ref.float()-e.y[off:off+c].float()).abs();diffs.append(dict(max_abs=d.max().item(),rms=d.square().mean().sqrt().item(),ref_rms=ref.float().square().mean().sqrt().item()))
 results.append(dict(rows=sizes,errors=diffs))
print(json.dumps(dict(torch=torch.__version__,triton=triton.__version__,gpu=torch.cuda.get_device_name(),results=results,workspace=e.workspace_bytes),indent=2))
