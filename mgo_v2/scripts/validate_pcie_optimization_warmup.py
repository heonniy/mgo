"""Record actual grouped and metadata preflight without treating warmup as timing."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from pcie_host import write

def run(root,out):
    rows={};sources={}
    for base in ('R-NEAR','G-NEAR'):
        ranks=[]
        for rank in range(4):
            paths=[root/(base+'__'+v) for v in ('NATIVE','GROUPED','GROUPED_META')]
            result=[json.loads((p/f'repeat0_rank{rank}.json').read_text()) for p in paths]
            assert all(r['phase']=='warmup' and r['output_tokens']==64 and r['finite_logits'] for r in result)
            assert all(r['validation']['status']=='PASS' and r['validation']['physical_slots']==1843 for r in result)
            tokens=[np.array(r['tokens']) for r in result]
            assert np.array_equal(tokens[0][:,0],tokens[1][:,0])
            assert np.array_equal(tokens[1],tokens[2])
            assert result[1]['validation']['state_hash']==result[2]['validation']['state_hash']
            assert np.array_equal(np.load(paths[1]/f'policy_trace_repeat0_rank{rank}.npy'),np.load(paths[2]/f'policy_trace_repeat0_rank{rank}.npy'))
            checks=[json.loads((p/f'grouped_repeat0_rank{rank}.json').read_text()) for p in paths[1:]]
            assert all(len(r['checks'])==48 and r['decode_calls']==3024 and r['all_ready'] for r in checks)
            assert all(x['finite'] and x['relative_l2']<=.01 for r in checks for x in r['checks'])
            meta=json.loads((paths[2]/f'metadata_repeat0_rank{rank}.json').read_text());assert meta['wire_checks']==48 and meta['calls']==3024
            ranks.append(dict(rank=rank,physical_gpu=[0,1,4,5][rank],native_grouped_first_token_exact=True,
                         native_grouped_token_agreement=float((tokens[0]==tokens[1]).mean()),
                         grouped_metadata_full_token_cache_trace_exact=True,metadata_wire_checks=48,
                         grouped_max_relative_l2=max(x['relative_l2'] for r in checks for x in r['checks']),
                         workspace_bytes=checks[0]['workspace_bytes'],finite_full64=True))
            for p in paths:
                for name in (f'repeat0_rank{rank}.json',f'policy_trace_repeat0_rank{rank}.npy',f'grouped_repeat0_rank{rank}.json',f'pinned_rank{rank}.json'):
                    path=p/name;sources[str(path)]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            path=paths[2]/f'metadata_repeat0_rank{rank}.json'
            sources[str(path)]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows[base]=ranks
    write(out,dict(status='PASS',source=str(root),scope='Disjoint full64 warmup preflight only; no primary performance claims',
                   cross_rank_physical_phase_validation='PENDING separate post-primary diagnostics',rows=rows,sources=sources))
    print(json.dumps(dict(status='PASS',out=str(out),wire_checks=384,actual_grouped_full64_generations=4)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
