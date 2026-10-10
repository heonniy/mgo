"""Construct a deterministic rank-locality upper-bound decode route.

Every rank's 16 real model hidden states select eight different experts from
that rank's disjoint 32-expert block. The prefill and model weights remain
unchanged. This is a synthetic router intervention, not a quality benchmark.
"""

import hashlib
import json
from pathlib import Path

import torch


SOURCE = Path('/home/hwlee/mgo-results/single_step_locality_pilot_20261010/jobs/r4_n_br_c30_16_diagnostic_v4')
OUTPUT = Path('/home/hwlee/mgo-results/single_step_locality_pilot_20261010/synthetic_routes')


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    digests=[]
    for rank in range(4):
        captured=torch.load(SOURCE/f'decode_routes_rank{rank}.pt',map_location='cpu',weights_only=True)
        routes=[];digest=hashlib.sha256()
        for layer,(selected,weights,probs) in enumerate(captured['routes']):
            assert selected.shape==(16,8) and probs.shape==(16,128)
            expert_ids=[rank*32+(layer*8+k)%32 for k in range(8)]
            selected=torch.tensor(expert_ids,dtype=selected.dtype).expand(16,-1).clone()
            weights=torch.full_like(weights,1/8)
            probs=torch.zeros_like(probs)
            probs.scatter_(1,selected,1/8)
            assert torch.allclose(probs.sum(1),torch.ones(16))
            routes.append((selected,weights,probs))
            digest.update(selected.to(torch.int16).numpy().tobytes())
        torch.save(dict(cell=captured['cell'],rank=rank,routes=routes),OUTPUT/f'decode_routes_rank{rank}.pt')
        digests.append(digest.hexdigest())
    (OUTPUT/'manifest.json').write_text(json.dumps(dict(status='FROZEN',
        source=str(SOURCE),physical_gpus=[0,1,4,5],local_batch=16,layers=48,
        route_sha256_by_rank=digests,construction='disjoint rank-specific 8-of-32 experts per layer; uniform top8 weights'),indent=2)+'\n')
    print(OUTPUT)


if __name__=='__main__':main()
