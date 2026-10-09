"""CPU preflight: lossless compact current-routing replay on real G trace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import mgo_v2
from env_offload_policy import Policy
from pcie_host import write


def main(root,out):
    p=Policy([459,459,459,458],np.zeros((48,128,128),np.float32),False,7,42,
             native_pcie=True,quota_mode='group_balanced')
    source=root/'decision_capture_repeat-1.npz'
    with np.load(source) as stream:
        raw={name:stream[name] for name in ('selected','weights','origins','gate','offsets')}
        for event in range(768):
            lo,hi=raw['offsets'][event:event+2]
            selected=raw['selected'][lo:hi].astype(np.uint8).astype(np.int64)
            weights=raw['weights'][lo:hi].astype(np.float32)
            origins=raw['origins'][lo:hi].astype(np.int8).astype(np.int64)
            p.apply(event,selected,weights,origins,raw['gate'][event].astype(np.float32),None)
        np.testing.assert_array_equal(p.slots,stream['slots'][768])
    np.testing.assert_array_equal(p.native_pcie.trace(),np.load(root/'policy_trace_repeat-1_rank0.npy')[:768])
    import torch
    assert not torch.cuda.is_initialized()
    receipt=dict(status='PASS',events=768,source_capture=str(source),
          source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          actual_full_61_trace_exact=True,cold_cache_replay_state_exact=True,
          selected_uint8_weights_fp32_origins_int8_roundtrip_exact=True,cuda_initialized=False,
          scope='Real historical G routing validates nomination dtypes and independent-cache native replay; no serving timing',
          repaired_preflight_error='Direct env_offload_policy import produced a circular package import; initialize mgo_v2 before importing the adapter')
    write(out,receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
