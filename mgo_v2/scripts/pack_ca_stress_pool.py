"""Audit old/new exact traces and normalize request-addressable, readonly inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,numpy as np
from ca_stress_common import *

def main():
 for name in ['MATH','ShareGPT']:
  manifest=json.loads((ROOT/(name+'_requests.json')).read_text());requests=manifest['requests'];n=len(requests);dest=ROOT/'pool'/name
  if (dest/'receipt.json').exists():
   receipt=json.loads((dest/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['manifest_sha256']==sha(ROOT/(name+'_requests.json'));continue
  dest.mkdir(parents=True,exist_ok=False);lengths=np.array([len(r['input_ids']) for r in requests],np.int32);offsets=np.r_[0,np.cumsum(lengths,dtype=np.int64)];np.save(dest/'lengths.npy',lengths);np.save(dest/'offsets.npy',offsets)
  arrays={}
  shapes={'decode_selected':((256,48,n,8),'uint8'),'decode_weights':((256,48,n,8),'float32'),'decode_router':((256,48,n,128),'float32'),'prefill_selected':((48,int(offsets[-1]),8),'uint8'),'prefill_weights':((48,int(offsets[-1]),8),'float32'),'prefill_router_tail':((48,n,128,128),'float32'),'generated_tokens':((n,256),'int64')}
  for key,(shape,dtype) in shapes.items():arrays[key]=np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
  shards=[SOURCE/'captures'/name/f'rank{rank}' for rank in range(8)]
  shards += [wave/'captures'/name/f'rank{rank}' for wave in sorted((ROOT/'waves').glob('wave*')) if (wave/(name+'_requests.json')).exists() for rank in range(8)]
  seen=[];proof=[]
  for path in shards:
   receipt=json.loads((path/'receipt.json').read_text());assert receipt['status']=='PASS' and receipt['decode_forwards']==256
   ids=receipt['request_ids'];seen.extend(ids)
   for f in receipt['files']:assert sha(path/f['file'])==f['sha256']
   proof.append(dict(receipt=str(path/'receipt.json'),sha256=sha(path/'receipt.json'),requests=ids))
   if not ids:continue
   src={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in ['decode_selected','decode_weights','decode_router','prefill_selected','prefill_weights','prefill_router','generated_tokens','prefill_input_ids','prefill_active_mask']}
   assert src['decode_selected'].shape==(256,48,len(ids),8) and src['generated_tokens'].shape==(len(ids),256)
   for key in ['decode_selected','decode_weights','decode_router']:arrays[key][:,:,ids,:]=src[key]
   arrays['generated_tokens'][ids]=src['generated_tokens'];source_start=0
   for local,req in enumerate(ids):
    length=int(lengths[req]);assert np.array_equal(src['prefill_input_ids'][local,-length:],requests[req]['input_ids']);assert int(src['prefill_active_mask'][local].sum())==length
    lo,hi=offsets[req:req+2];end=source_start+length
    for key in ['prefill_selected','prefill_weights']:arrays[key][:,lo:hi]=src[key][:,source_start:end]
    tail=min(length,128);arrays['prefill_router_tail'][:,req,:tail]=src['prefill_router'][:,end-tail:end]
    source_start=end
   assert source_start==src['prefill_selected'].shape[1]
   for a in arrays.values():a.flush()
   del src
  assert sorted(seen)==list(range(n)) and len(seen)==len(set(seen))
  proxy=np.empty((48,n,8),np.uint8)
  events=[]
  for layer in range(48):
   step=(layer%4)*85;proxy[layer]=arrays['decode_selected'][step,layer];events.append(dict(step=step,layer=layer))
  np.save(dest/'proxy_routes.npy',proxy)
  receipt=dict(status='PASS',dataset=name,pool_size=n,manifest_sha256=sha(ROOT/(name+'_requests.json')),source_receipts=proof,original512_reused=True,decode_steps=256,proxy_events=events,files=[dict(name=p.name,sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.glob('*.npy'))])
  write(dest/'receipt.json',receipt);write(PACKET/(name+'_pool_validation.json'),receipt)
  print(name,n,'exact request-addressable pool validated',flush=True)
if __name__=='__main__':main()
