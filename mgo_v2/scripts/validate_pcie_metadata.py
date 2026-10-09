"""CPU-only differential gate for native rolling history and route counts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mgo_v2.eviction import GateHistory
from mgo_v2.pcie_native_metadata import NativeGateHistory,SOURCE,FLAGS

def main(output):
    rng=np.random.default_rng(730);checks=0
    for dtype in (np.float32,np.float64):
        reference=GateHistory(3,128,128);native=NativeGateHistory(3,128,128)
        for tokens in [0,1,64,128,512,16,64,64,16,129,7]*8:
            layer=checks%3;probs=rng.random((tokens,128)).astype(dtype)
            reference.update(layer,probs);native.update(layer,probs)
            assert np.array_equal(reference.sums,native.sums)
            assert np.array_equal(np.asarray([reference.score(layer,e) for e in range(128)],np.float32),native.gates(layer))
            assert len(reference.rows[layer])==len(native.rows[layer]);checks+=1
        ids=rng.integers(0,128,(64,8),dtype=np.uint8);hist=np.empty((4,128),np.int64)
        assert not native.native.mgo_route_histogram(ids.ctypes.data,hist.ctypes.data,4,16,8,128)
        assert np.array_equal(hist,np.stack([np.bincount(x.ravel(),minlength=128) for x in ids.reshape(4,16,8)]))
    result=dict(status='PASS',gate_updates=checks,exact_fp64_sums=True,exact_fp32_scores=True,
                exact_histograms=2,seed=730,source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),compiler_flags=FLAGS)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
