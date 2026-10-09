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
  e['targets']=reserve(e['targets']);e['selected']=reserve(e['selected'])
  if 'coslot_send_idx' in e:
   e['coslot_combine']=[(reserve(idx),reserve(pos)) for idx,pos in e['coslot_combine']]
   e['coslot_send_idx']=reserve(e['coslot_send_idx']);e['coslot_send_eids']=reserve(e['coslot_send_eids'])
   e['coslot_groups']=[(expert,reserve(rows),slot) for expert,rows,slot in e['coslot_groups']]
   e['coslot_return_order']=reserve(e['coslot_return_order'])
  layouts.append(e)
 storage=torch.from_numpy(np.concatenate(pieces) if pieces else np.empty(0,np.int64)).to(device);pieces.clear()
 def view(spec):
  offset,length,shape=spec;x=storage[offset:offset+length]
  return x if len(shape)==1 else x.view(shape)
 for e in layouts:
  for key in ['send_idx','send_eids','return_order','targets','selected']:e[key]=view(e[key])
  e['groups']=[(expert,view(rows),view(cols),slot) for expert,rows,cols,slot in e['groups']]
  e['combine']=[(view(idx),view(pos)) for idx,pos in e['combine']]
  if 'coslot_send_idx' in e:
   e['coslot_combine']=[(view(idx),view(pos)) for idx,pos in e['coslot_combine']]
   e['coslot_send_idx']=view(e['coslot_send_idx']);e['coslot_send_eids']=view(e['coslot_send_eids'])
   e['coslot_groups']=[(expert,view(rows),slot) for expert,rows,slot in e['coslot_groups']]
   e['coslot_return_order']=view(e['coslot_return_order'])
 return layouts


def pack_rank_partial_layout(event,device='cuda'):
 """One contiguous index transfer; no Python-list round trip or unused returns."""
 if not event.get('rank_partial_layout'):raise ValueError('rank-partial layout required')
 pieces=[]
 def reserve(array):
  array=np.asarray(array,dtype=np.int64)
  index=len(pieces);pieces.append(array.reshape(-1));return index
 send_idx=reserve(event['send_idx']);send_eids=reserve(event['send_eids']);targets=reserve(event['targets'])
 groups=[(expert,reserve(rows),reserve(cols),slot) for expert,rows,cols,slot in event['groups']]
 packed=np.concatenate(pieces)
 storage=torch.from_numpy(packed).to(device)
 # One C++ split creates the views in packet order; slicing each group from
 # Python used two tensor operations per expert, per layer, per token.
 views=storage.split([piece.size for piece in pieces])
 result=dict(event,send_idx=views[send_idx],send_eids=views[send_eids].view(event['send_eids'].shape),targets=views[targets],groups=[(expert,views[rows],views[cols],slot) for expert,rows,cols,slot in groups],layout_index_bytes=int(packed.nbytes))
 return result
