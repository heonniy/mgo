"""Pack immutable indices into one device allocation outside measurement."""
import numpy as np
import torch

def pack_layouts(events,device='cuda'):
 pieces=[];size=0
 def reserve(x,shape=None):
  nonlocal size
  a=np.asarray(x,dtype=np.int64)
  if shape is not None:a=a.reshape(shape)
  spec=(size,a.size,a.shape);size+=a.size;pieces.append(a.reshape(-1));return spec
 layouts=[]
 for event in events:
  e=dict(event)
  e['send_idx']=reserve(e['send_idx']);e['send_eids']=reserve(e['send_eids'],(-1,8))
  e['groups']=[(expert,reserve(rows),reserve(cols),slot) for expert,rows,cols,slot in e['groups']]
  e['return_order']=reserve(e['return_order']);e['combine']=[(reserve(idx),reserve(pos)) for idx,pos in e['combine']]
  e['targets']=reserve(e['targets']);e['selected']=reserve(e['selected']);layouts.append(e)
 storage=torch.from_numpy(np.concatenate(pieces) if pieces else np.empty(0,np.int64)).to(device);pieces.clear()
 def view(spec):
  offset,length,shape=spec;x=storage[offset:offset+length]
  return x if len(shape)==1 else x.view(shape)
 for e in layouts:
  for key in ['send_idx','send_eids','return_order','targets','selected']:e[key]=view(e[key])
  e['groups']=[(expert,view(rows),view(cols),slot) for expert,rows,cols,slot in e['groups']]
  e['combine']=[(view(idx),view(pos)) for idx,pos in e['combine']]
 return layouts
