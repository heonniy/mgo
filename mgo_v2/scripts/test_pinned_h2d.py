import torch
from mgo_v2.pinned_h2d import copy_expert_to_stage
def test_copy_expert_to_stage_cpu_layout():
 stage=torch.empty(6,dtype=torch.float32);tensors=[torch.tensor([1.,2.]),torch.tensor([[3.,4.],[5.,6.]])]
 assert copy_expert_to_stage(stage,tensors)==6
 assert stage.tolist()==[1.,2.,3.,4.,5.,6.]
